#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <unordered_set>

#include <cub/cub.cuh>

#include "generate.hpp"

inline constexpr auto warp_size = 32;
inline constexpr auto block_size = 128;

extern __shared__ char shared_memory[];

// `packed_values` is an array: [packed_values, packed_values + nnz)
// `filled` is an array of bits: [filled, filled + ceil(n / bits_per_word))
// Unpack up to n <= block_size elements from `packed_values`, storing them
// at locations determined by `filled`.
template <typename T, typename I>
__device__ auto sparse_iteration_block(T* packed_values, I* filled, T* unpacked, std::size_t n) {
  auto tid = threadIdx.x;

  using indices_type = int;

  auto bits_per_word = sizeof(I)*8;

  indices_type* indices = (indices_type*) shared_memory;

  indices[tid] = 0;
  bool has_value;
  // indices[idx] = idx'th set bit of `filled`
  if (tid < n) {
    auto idx = tid;

    // Check the idx'th bit of `filled`
    auto element = idx / bits_per_word;
    auto bit = idx % bits_per_word;
    has_value = (0x1 << bit) & filled[element];

    if (has_value) {
      indices[idx] = 1;
    }
  }

  // Perform prefix_sum on [indices, indices + n)
  // Specialize BlockScan for a 1D block of 128 threads of type int
  typedef cub::BlockScan<indices_type, block_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  BlockScan(temp_storage).ExclusiveSum(indices[tid], indices[tid]);

  T* values = (T*) (indices + block_size);

  if (tid < n) {
    if (has_value) {
      values[tid] = packed_values[indices[tid]];
      unpacked[tid] = packed_values[indices[tid]];
    } else {
      values[tid] = 0;
      unpacked[tid] = 0;
    }
  }

  auto idx = n - 1;

  bool last_filled = (0x1 << (idx % bits_per_word)) & filled[idx / bits_per_word];

  __syncthreads();

  return (indices[n-1] - indices[0]) + last_filled;
}

template <typename T, typename I>
__device__ void sparse_iteration(T* packed_values, I* filled, T* unpacked, std::size_t n) {
  constexpr bool print = false;
  constexpr auto bits_per_word = sizeof(I)*8;

  std::size_t iteration = 0;
  std::size_t vals_consumed = 0;
  for (std::size_t filled_position = 0; filled_position < n; filled_position += block_size) {
    auto block_unpacked = unpacked + filled_position;
    auto consumed = sparse_iteration_block(packed_values + vals_consumed, filled + (filled_position / bits_per_word), block_unpacked, min(block_size, int(n - filled_position)));
    vals_consumed += consumed;

    if (print && threadIdx.x == 0) {
      printf("(Block %d) Iteration %lu: %d values consumed\n", blockIdx.x, iteration, consumed);
    }

    ++iteration;
  }
}

template <typename T, typename I>
__global__ void sparse_iteration(T* packed_values, I* filled, I* block_offsets, T* unpacked, std::size_t n, std::size_t words_per_block) {
  constexpr auto bits_per_word = sizeof(I)*8;
  auto values_offset = block_offsets[blockIdx.x];
  auto filled_offset = words_per_block*blockIdx.x;
  auto block_unpacked = unpacked + filled_offset*bits_per_word;
  sparse_iteration(packed_values + values_offset, filled + filled_offset, block_unpacked, min(words_per_block*bits_per_word, n - blockIdx.x*words_per_block));
}

int main(int argc, char** argv) {
  std::size_t m = 10000;
  std::size_t n = 10000;
  std::size_t nnz = 0.5 * m*n;

  printf("Generating matrix with %lu nnz (%lf%% filled)\n", nnz, 100*(double(nnz) / (m*n)));

  using T = float;
  using I = int;

  // Generate sparse matrix.
  auto&& [values, rowptr, colind, shape, _] = generate_csr<T, I>(m, n, nnz);

  // Analyze sparse matrix to ensure no duplicate column indices.
  std::size_t counted_nnz = 0;
  std::size_t duplicate_nnz = 0;
  spa_set<I> column_indices(n);
  for (I i = 0; i < m; i++) {
    column_indices.clear();
    for (I j_ptr = rowptr[i]; j_ptr < rowptr[i+1]; j_ptr++) {
      counted_nnz++;
      if (column_indices.contains(colind[j_ptr])) {
        duplicate_nnz++;
      }
    }
  }

  printf("Counted %lu NNZ, %lu duplicate column indices\n", counted_nnz, duplicate_nnz);

  // Generate `filled` bitarray with nonzero pattern.

  auto bits_per_word = sizeof(I)*8;

  auto num_words = (m*n + bits_per_word - 1) / bits_per_word;

  std::vector<I> filled(num_words, I(0));

  for (I i = 0; i < m; i++) {
    for (I j_ptr = rowptr[i]; j_ptr < rowptr[i+1]; j_ptr++) {
      auto j = colind[j_ptr];
      auto v = values[j_ptr];
      auto idx = i * n + j;

      // Write to the idx'th bit of `filled`
      auto element = idx / bits_per_word;
      auto bit = idx % bits_per_word;
      filled[element] |= (0x1 << bit);
    }
  }

  // Ensure that the filled array has the correct number of nonzeros.

  std::size_t num_elements = 0;
  for (std::size_t idx = 0; idx < m*n; idx++) {
    auto element = idx / bits_per_word;
    auto bit = idx % bits_per_word;
    bool present = filled[element] & (0x1 << bit);
    if (present) {
      num_elements++;
    }
  }

  assert(num_elements == nnz);

  // Generate local version of the unpacked array.
  std::vector<T> unpacked(m*n, 0);

{
  std::size_t count = 0;
  for (I idx = 0; idx < m*n; idx++) {
    auto element = idx / bits_per_word;
    auto bit = idx % bits_per_word;
    if (filled[element] & (0x1 << bit)) {
      unpacked[idx] = values[count++];
    }
  }
}

  std::size_t num_blocks = 200;

  // Generate array of offsets at which each block should start.

  std::size_t words_per_block = (filled.size() + num_blocks - 1) / num_blocks;
  std::vector<I> nnz_per_block(num_blocks, 0);

  for (std::size_t block = 0; block < num_blocks; block++) {
    for (std::size_t i = words_per_block*block; i < std::min(filled.size(), words_per_block*(block+1)); i++) {
      nnz_per_block[block] += __builtin_popcount(filled[i]);
    }
  }

  std::exclusive_scan(nnz_per_block.begin(), nnz_per_block.end(), nnz_per_block.begin(), std::size_t(0));

  printf("Offsets for each block: ");
  for (std::size_t i = 0; i < nnz_per_block.size(); i++) {
    printf("%d ", nnz_per_block[i]);
  }
  printf("\n");

  printf("Each block will process %lu bits of `filled` (~%lu iterations)\n",
         words_per_block*bits_per_word, words_per_block / block_size);

  // Allocate and copy data to GPU.

  T* a_d;
  I* filled_d;
  I* nnz_per_block_d;
  T* unpacked_d;
  cudaMalloc(&a_d, nnz*sizeof(T));
  cudaMalloc(&filled_d, num_words*sizeof(I));
  cudaMalloc(&nnz_per_block_d, num_blocks*sizeof(I));
  cudaMallocManaged(&unpacked_d, m*n*sizeof(T));
  cudaMemcpy(a_d, values.data(), nnz*sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(filled_d, filled.data(), num_words*sizeof(I), cudaMemcpyHostToDevice);
  cudaMemcpy(nnz_per_block_d, nnz_per_block.data(), num_blocks*sizeof(I), cudaMemcpyHostToDevice);
  cudaMemset(unpacked_d, 0, m*n*sizeof(T));
  cudaDeviceSynchronize();

  // Launch unpacking kernel on GPU.
  auto begin = std::chrono::high_resolution_clock::now();
  sparse_iteration<<<num_blocks, block_size, block_size * (sizeof(T) + sizeof(I))>>>(a_d, filled_d, nnz_per_block_d, unpacked_d, m*n, words_per_block);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double mbytes = nnz*sizeof(T) * 1e-6;
  double gbytes = nnz*sizeof(T) * 1e-9;

  printf("Took %lf ms to decompress %lf MB\n", duration*1000, mbytes);
  printf("%lf GB/s\n", gbytes / duration);

  // Compare to original sparse matrix, ensure correct unpacking.

  printf("Comparing to original sparse matrix...\n");

  std::size_t nonzero_pattern_incorrect = 0;
  std::size_t explicit_zeros = 0;
  std::size_t incorrect_values = 0;
  for (I i = 0; i < m; i++) {
    for (I j_ptr = rowptr[i]; j_ptr < rowptr[i+1]; j_ptr++) {
      auto j = colind[j_ptr];
      auto v = values[j_ptr];
      auto unpacked_value = unpacked_d[i*n + j];

      if (v != unpacked_value) {
        incorrect_values++;
      }

      if (v == 0) {
        explicit_zeros++;
      }

      if (unpacked_value == 0 && v != 0) {
        nonzero_pattern_incorrect++;
      }
    }
  }

  std::size_t zeros_count = 0;
  for (std::size_t i = 0; i < m*n; i++) {
    if (unpacked_d[i] == 0) {
      zeros_count++;
    }
  }

  if (nonzero_pattern_incorrect == 0 && zeros_count == m*n - nnz + explicit_zeros) {
    printf("Nonzero pattern OK.\n");
  } else {
    printf("ERROR: nonzero pattern INCORRECT.\n");
  }

  if (incorrect_values == 0) {
    printf("Values OK!\n");
  } else {
    printf("ERROR: %lu values INCORRECT.\n", incorrect_values);
  }

  std::size_t count = 0;
  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j < n; j++) {
      if (unpacked_d[i * n + j] != 0) {
        count++;
      }
    }
  }
  printf("%lu nnz\n", count);

  std::size_t wrong_values = 0;
  for (std::size_t i = 0; i < m*n; i++) {
    if (unpacked_d[i] != unpacked[i]) {
      wrong_values++;
    }
  }

  if (wrong_values != 0) {
    printf("ERROR: %lu/%lu values INCORRECT compared to CPU unpacked data.\n", wrong_values, m*n);
  } else {
    printf("Values OK compared to CPU unpacked data.\n");
  }

  return 0;
}
