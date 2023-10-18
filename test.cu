#include <cuda.h>
#include <cstdint>
#include <cstdio>

#include <cub/cub.cuh>

#include "generate.hpp"

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline constexpr auto warp_size = 32;

inline constexpr auto block_size = 128;

extern __shared__ char shared_memory[];

template <typename T, typename I>
__device__ auto sparse_iteration_block(T* packed_values, I* filled, std::size_t num_values) {
  auto tid = threadIdx.x;

  using indices_type = int;

  indices_type* indices = (indices_type*) shared_memory;

  // indices[idx] = idx'th set bit of `filled`
  if (tid < num_values) {
    auto idx = tid;

    // Check the idx'th bit of `filled`
    auto element = idx / (sizeof(std::uint64_t)*8);
    auto bit = idx % (sizeof(std::uint64_t)*8);
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

  if (tid < num_values) {
    BlockScan(temp_storage).ExclusiveSum(indices[tid], indices[tid]);
  } else {
    int dummy_value = 0;
    BlockScan(temp_storage).ExclusiveSum(dummy_value, dummy_value);
  }

  T* values = (T*) (indices + num_values);

  if (tid < num_values) {
    values[tid] = packed_values[indices[tid]];
  }

  return indices[num_values-1] - indices[0];
}

// Written for a single warp
template <typename T, typename I>
__global__ void sparse_iteration(T* packed_values, I* filled, std::size_t num_values) {
  if (blockIdx.x == 0) {
    auto consumed = sparse_iteration_block(packed_values, filled, block_size);
    if (threadIdx.x == 0)
      printf("Consumed %d values\n", consumed);
  }
}

int main(int argc, char** argv) {
  std::size_t m = 1000;
  std::size_t n = 1000;
  std::size_t nnz = m*n / 2;

  using T = float;
  using I = int;

  auto&& [values, rowptr, colind, shape, _] = generate_csr<T, I>(m, n, nnz);

  auto bits_per_word = sizeof(I)*8;

  auto num_words = (m*n + bits_per_word - 1) / bits_per_word;

  std::vector<T> a;
  std::vector<I> filled(num_words, I(0));
  a.reserve(nnz);

  for (int i = 0; i < m; i++) {
    for (int j_ptr = rowptr[i]; j_ptr < rowptr[i+1]; j_ptr++) {
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

  T* a_d;
  I* filled_d;
  CHECK_CUDA(cudaMalloc(&a_d, m*n*sizeof(T)));
  CHECK_CUDA(cudaMalloc(&filled_d, num_words*sizeof(I)));
  CHECK_CUDA(cudaMemcpy(a_d, a.data(), m*n*sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(filled_d, filled.data(), num_words*sizeof(I), cudaMemcpyHostToDevice));

  cudaDeviceSynchronize();

  sparse_iteration<<<1, block_size, block_size * (sizeof(T) + sizeof(I))>>>(a_d, filled_d, nnz);
  cudaDeviceSynchronize();

  CHECK_CUDA(cudaGetLastError());

  return 0;
}
