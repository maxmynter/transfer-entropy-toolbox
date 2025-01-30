#include "discrete_multivar_entropy.h"
#include <numeric>
#include <stdexcept>

namespace fast_entropy {

double discrete_multivar_joint_entropy(
    const std::vector<py::array_t<int64_t>> &classes,
    const std::vector<int> &n_classes) {

  if (classes.empty() || n_classes.empty()) {
    throw std::invalid_argument("Empty input arrays");
  }

  // Get size of first array to determine n_steps
  auto first_buf = classes[0].request();
  const size_t n_steps = first_buf.size;

  // Calculate total size of histogram
  size_t total_bins = std::accumulate(n_classes.begin(), n_classes.end(), 1ULL,
                                      std::multiplies<size_t>());

  // Create histogram array
  std::vector<size_t> hist(total_bins, 0);

  // Get raw pointers to data
  std::vector<const int64_t *> data_ptrs;
  for (const auto &arr : classes) {
    auto buf = arr.request();
    if (buf.size != n_steps) {
      throw std::invalid_argument("All input arrays must have same length");
    }
    data_ptrs.push_back(static_cast<const int64_t *>(buf.ptr));
  }

  // Fill histogram
  for (size_t i = 0; i < n_steps; ++i) {
    // Calculate index in flattened histogram
    size_t idx = 0;
    size_t stride = 1;

    for (size_t j = 0; j < classes.size(); ++j) {
      const int64_t val = data_ptrs[j][i];
      if (val < 0 || val >= n_classes[j]) {
        throw std::out_of_range("Class index out of bounds");
      }
      idx += val * stride;
      stride *= n_classes[j];
    }

    hist[idx]++;
  }

  // Calculate entropy
  double entropy = 0.0;
  const double n_steps_inv = 1.0 / static_cast<double>(n_steps);

  for (const auto &count : hist) {
    if (count > 0) {
      const double p = static_cast<double>(count) * n_steps_inv;
      entropy -= p * std::log(p);
    }
  }

  return entropy;
}
} // namespace fast_entropy
