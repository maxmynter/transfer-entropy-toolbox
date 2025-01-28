#include "discrete_conditional_entropy.h"
#include <stdexcept>

namespace fast_entropy {

double discrete_conditional_entropy(const py::array_t<int64_t> &data,
                                    const std::vector<int> &n_classes) {
  // Conditional entropy H(Xi|Xj): H(Xi,Xj) - H(Xj)
  py::buffer_info buf = data.request();

  // Input validation
  if (buf.ndim != 2) {
    throw std::runtime_error(
        "Need 2D array for conditional entropy, [timesteps x 2]");
  }
  if (buf.shape[1] != 2) {
    throw std::runtime_error("Array must have 2 columns.");
  }
  if (n_classes.size() != 2) {
    throw std::runtime_error("n_classes must have 2 elements. The number of "
                             "classes per time series.");
  }

  const size_t n_steps = buf.shape[0];
  const int64_t *ptr = static_cast<int64_t *>(buf.ptr);

  double joint_ent = discrete_joint_entropy(data, n_classes);

  size_t row_stride = buf.strides[0] / sizeof(int64_t);
  const int COLUMN_XJ = 1;

  std::vector<int64_t> xj_data(n_steps);
  for (size_t i = 0; i < n_steps; ++i) {
    xj_data[i] = ptr[i * row_stride + COLUMN_XJ];
  }

  py::array_t<int64_t> xj_array(n_steps, xj_data.data());
  double xj_ent = discrete_entropy(xj_array, n_classes[COLUMN_XJ]);

  return joint_ent - xj_ent;
}

} // namespace fast_entropy
