#include "discrete_joint_entropy.h"
#include <cstddef>
#include <stdexcept>

namespace fast_entropy {
double discrete_joint_entropy(const py::array_t<int64_t> &data,
                              const std::vector<int> &n_classes) {
py:
  pybind11::buffer_info buf = data.request();

  if (buf.ndim != 2) {
    throw std::runtime_error(
        "Need 2D array for bivariate joint entropy, [timesteps x 2]");
  }
  if (buf.shape[1] != 2) {
    throw std::runtime_error("Array must have 2 columns.");
  }

  if (n_classes.size() != 2) {
    throw std::runtime_error("n_classes must have 2 elements. The number of "
                             "classes per time series.");
  }

  const size_t n_steps = buf.shape[0];
  const size_t hist_size =
      static_cast<size_t>(n_classes[0] * static_cast<size_t>(n_classes[1]));
  std::vector<size_t> hist(hist_size, 0);

  const int64_t *ptr = static_cast<int64_t *>(buf.ptr);

  for (size_t step = 0; step < n_steps; ++step) {
    const int64_t val_0 = ptr[step * 2];
    const int64_t val_1 = ptr[step * 2 + 1];

    hist[val_0 * n_classes[1] + val_1]++;
  }

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
