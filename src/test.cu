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
// `filled` is an array of bits: [filled, filled + n)

template <typename T, typename I>
__device__ auto sparse_iteration_block(T* packed_values, I* filled, std::size_t n) {
  auto tid = threadIdx.x;

  using indices_type = int;

  auto bits_per_word = sizeof(I)*8;

  indices_type* indices = (indices_type*) shared_memory;

  // indices[idx] = idx'th set bit of `filled`
  if (tid < n) {
    auto idx = tid;

    // Check the idx'th bit of `filled`
    auto element = idx / bits_per_word;
    auto bit = idx % bits_per_word;
    bool has_value = (0x1 << bit) & filled[element];

    if (has_value) {
      indices[idx] = 1;
    } else {
      indices[idx] = 0;
    }
  }

  // Perform prefix_sum on [indices, indices + num_values)
  // Specialize BlockScan for a 1D block of 128 threads of type int
  typedef cub::BlockScan<I, block_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  if (tid < n) {
    BlockScan(temp_storage).ExclusiveSum(indices[tid], indices[tid]);
  } else {
    int dummy_value = 0;
    BlockScan(temp_storage).ExclusiveSum(dummy_value, dummy_value);
  }

  T* values = (T*) (indices + n);

  if (tid < n) {
    values[tid] = packed_values[indices[tid]];
  }

  auto idx = block_size - 1;

  bool last_filled = (0x1 << (idx % bits_per_word)) & filled[idx / bits_per_word];

  return (indices[n-1] - indices[0]) + last_filled;
}

template <typename T, typename I>
__global__ void sparse_iteration(T* packed_values, I* filled, std::size_t n) {
  constexpr bool print = false;
  auto bits_per_word = sizeof(I)*8;
  if (blockIdx.x == 0) {
    std::size_t n_consumed = 0;
    std::size_t vals_consumed = 0;
    std::size_t iteration = 0;
    while (n_consumed < n) {
      auto consumed = sparse_iteration_block(packed_values + vals_consumed, filled + (n_consumed / bits_per_word), block_size);
      vals_consumed += consumed;
      n_consumed += block_size;

      if (print && threadIdx.x == 0) {
        printf("Iteration %lu: %d values consumed\n", iteration, consumed);
      }

      ++iteration;
    }
  }
}

int main(int argc, char** argv) {
  std::size_t m = 40000;
  std::size_t n = 40000;
  std::size_t nnz = 0.5 * m*n;

  printf("Generating matrix with %lu nnz (%lf%% filled)\n", nnz, 100*(double(nnz) / (m*n)));

  using T = float;
  using I = int;

  auto&& [values, rowptr, colind, shape, _] = generate_csr<T, I>(m, n, nnz);

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

  auto bits_per_word = sizeof(I)*8;

  auto num_words = (m*n + bits_per_word - 1) / bits_per_word;

  std::vector<T> a;
  std::vector<I> filled(num_words, I(0));
  a.reserve(nnz);

  for (I i = 0; i < m; i++) {
    for (I j_ptr = rowptr[i]; j_ptr < rowptr[i+1]; j_ptr++) {
      auto j = colind[j_ptr];
      auto v = values[j_ptr];
      a.push_back(v);
      auto idx = i * n + j;

      // Write to the idx'th bit of `filled`
      auto element = idx / bits_per_word;
      auto bit = idx % bits_per_word;
      filled[element] |= (0x1 << bit);
    }
  }


  std::size_t num_elements = 0;
  for (std::size_t idx = 0; idx < m*n; idx++) {
    auto element = idx / bits_per_word;
    auto bit = idx % bits_per_word;
    bool present = filled[element] & (0x1 << bit);
    if (present) {
      num_elements++;
    }
  }

  printf("Counted %lu num_elements\n", num_elements);

  T* a_d;
  I* filled_d;
  cudaMalloc(&a_d, nnz*sizeof(T));
  cudaMalloc(&filled_d, num_words*sizeof(I));
  cudaMemcpy(a_d, a.data(), nnz*sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(filled_d, filled.data(), num_words*sizeof(I), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  auto begin = std::chrono::high_resolution_clock::now();
  sparse_iteration<<<1, block_size, block_size * (sizeof(T) + sizeof(I))>>>(a_d, filled_d, nnz);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double mbytes = nnz*sizeof(T) * 1e-6;

  printf("Took %lf ms to decompress %lf MB\n", duration*1000, mbytes);
  printf("%lf GB/s\n", mbytes / duration);

  std::size_t block_id = 0;
  std::size_t total_count = 0;
  for (std::size_t i = 0; i < m*n; i += block_size) {
    std::size_t count = 0;
    for (std::size_t j = i; j < std::min(m*n, i+block_size); j++) {
      // Write to the idx'th bit of `filled`
      auto element = j / bits_per_word;
      auto bit = j % bits_per_word;
      bool has_value = filled[element] & (0x1 << bit);
      if (has_value) {
        count++;
        total_count++;
      }
    }
    // printf("Block %lu has %lu/%d values\n", block_id, count, block_size);
    block_id++;
    // if (block_id > 10)
    //  break;
  }

  assert(total_count == nnz);

  return 0;
}
