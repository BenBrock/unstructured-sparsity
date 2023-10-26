
#include <vector>
#include <tuple>
#include <random>

template <std::integral I>
class spa_set {
public:
  spa_set(I size) : data_(size, false) {}

  void insert(I key) {
    data_[key] = true;
    indices_.push_back(key);
  }

  bool contains(I key) const {
    return data_[key];
  }

  void clear() {
    for (auto&& index : indices_) {
      data_[index] = false;
    }
    indices_.clear();
  }

private:
  std::vector<bool> data_;
  std::vector<I> indices_;
};

template <typename T = float, typename I = int>
auto generate_csr(I m, I n, std::size_t nnz, std::size_t seed = 0) {
  std::vector<T> values;
  std::vector<I> colind;

  values.reserve(nnz);
  colind.reserve(nnz);

  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, n - 1);
  std::uniform_real_distribution d_f(0.0, 100.0);
  std::uniform_int_distribution<I> d_m(0, nnz);
  std::uniform_int_distribution<I> d_i(0, m - 1);

  for (std::size_t i = 0; i < nnz; i++) {
    values.push_back(d_f(g));
  }

  std::vector<I> rowptr;
  rowptr.reserve(m + 1);
  rowptr.push_back(0);

  // Generate a list of random numbers that sums to nnz.
  // This will contain the number of nonzeros in each row.
  for (std::size_t i = 0; i < m - 1; i++) {
    rowptr.push_back(d_m(g));
  }
  rowptr.push_back(nnz);

  std::sort(rowptr.begin(), rowptr.end());

  for (std::size_t i = m; i >= 1; i--) {
    rowptr[i] -= rowptr[i - 1];
  }

  for (auto&& v : rowptr) {
    v = std::min(v, n);
  }

  auto sum = std::reduce(rowptr.begin(), rowptr.end());

  while (sum < nnz) {
    auto row = d_i(g);

    rowptr[row] = std::min(I(rowptr[row] + (nnz - sum)), n);

    sum = std::reduce(rowptr.begin(), rowptr.end());
  }

  // Perform a prefix sum on this list to compute the final rowptrs array.
  std::inclusive_scan(rowptr.begin(), rowptr.end(), rowptr.begin());

  constexpr double sparsity_cutoff = 0.05;

  if (double(nnz) / (m*n) < sparsity_cutoff) {
    for (std::size_t i = 0; i < nnz; i++) {
      colind.push_back(d(g));
    }

    spa_set<I> column_indices(n);
    for (std::size_t i = 0; i < m; i++) {
      column_indices.clear();
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i+1]; j_ptr++) {
        // While the current column index has already occurred in this row,
        // keep generating a new column index until we have a unique one.
        while (column_indices.contains(colind[j_ptr])) {
          colind[j_ptr] = d(g);
        }

        column_indices.insert(colind[j_ptr]);
      }
    }
  } else {
    std::vector<I> v;
    v.reserve(n);
    for (I i = 0; i < n; i++) {
      v.push_back(i);
    }

    std::shuffle(v.begin(), v.end(), g);

    for (std::size_t i = 0; i < m; i++) {
      std::size_t row_size = rowptr[i+1] - rowptr[i];
      std::size_t starting_index = d(g);

      assert(row_size <= n);
      for (std::size_t j = 0; j < row_size; j++) {
        colind.push_back(v[(starting_index + j) % v.size()]);
      }
    }
  }

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
      column_indices.insert(colind[j_ptr]);
    }
  }

  return std::tuple(values, rowptr, colind, std::tuple<I, I>(m, n), I(nnz));
}
