#include "discrete_entropy.h"

namespace fast_entropy {

double discrete_entropy(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> data,
    int n_classes) {

  auto buf = data.request();
  const auto *ptr = static_cast<const int64_t *>(buf.ptr);
  const size_t n_steps = buf.size;

  // Use uint32_t for counts - more cache efficient than double for counting
  std::vector<uint32_t> local_counts(n_classes, 0);

  // First pass: count frequencies
  // No OpenMP here as for small arrays the overhead isn't worth it
  for (size_t i = 0; i < n_steps; ++i) {
    const auto idx = ptr[i];
    if (idx >= 0 && idx < n_classes) {
      local_counts[idx]++;
    }
  }

  // Second pass: compute entropy
  double entropy = 0.0;
  const double n_steps_inv = 1.0 / static_cast<double>(n_steps);

// This loop is worth parallelizing if n_classes is large enough
#pragma omp parallel for reduction(+ : entropy) if (n_classes > 100)
  for (int i = 0; i < n_classes; ++i) {
    if (local_counts[i] > 0) {
      const double p = static_cast<double>(local_counts[i]) * n_steps_inv;
      entropy -= p * std::log(p);
    }
  }

  return entropy;
}

} // namespace fast_entropy
