#include "discrete_transfer_entropy.h"
#include "discrete_entropy.h"
#include "discrete_joint_entropy.h"
#include "discrete_multivar_entropy.h"
#include <stdexcept>
#include <vector>

namespace fast_entropy {

double discrete_transfer_entropy(const py::array_t<int64_t> &data,
                                 const std::vector<int> &n_classes, int lag) {
  // Calculates TE(X->Y) for 2d array ([Y, X])
  py::buffer_info buf = data.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Input array must be 2-dimensional");
  }

  const size_t n_steps = buf.shape[0];
  if (n_steps <= lag) {
    throw std::runtime_error("Time series length must be greater than lag");
  }

  const size_t n_current = n_steps - lag;
  const int64_t *ptr = static_cast<const int64_t *>(buf.ptr);

  // Create arrays directly with shapes
  std::vector<ssize_t> shape_2d = {static_cast<ssize_t>(n_current), 2};
  py::array_t<int64_t> y_ylag(shape_2d);
  py::array_t<int64_t> ylag_xlag(shape_2d);

  std::vector<ssize_t> shape_1d = {static_cast<ssize_t>(n_current)};
  py::array_t<int64_t> ylag(shape_1d);
  py::array_t<int64_t> yt(shape_1d);
  py::array_t<int64_t> xt1(shape_1d);

  // Get mutable buffers
  auto y_ylag_buf = y_ylag.request();
  auto ylag_xlag_buf = ylag_xlag.request();
  auto ylag_buf = ylag.request();
  auto yt_buf = yt.request();
  auto xt1_buf = xt1.request();

  // Get pointers to the data
  int64_t *y_ylag_ptr = static_cast<int64_t *>(y_ylag_buf.ptr);
  int64_t *ylag_xlag_ptr = static_cast<int64_t *>(ylag_xlag_buf.ptr);
  int64_t *ylag_ptr = static_cast<int64_t *>(ylag_buf.ptr);
  int64_t *yt_ptr = static_cast<int64_t *>(yt_buf.ptr);
  int64_t *xt1_ptr = static_cast<int64_t *>(xt1_buf.ptr);

  // Fill all arrays in a single pass through the data
  for (size_t i = 0; i < n_current; ++i) {
    const int64_t current_y = ptr[(i + lag) * 2];
    const int64_t lagged_y = ptr[i * 2];
    const int64_t lagged_x = ptr[i * 2 + 1];

    y_ylag_ptr[i * 2] = current_y;
    y_ylag_ptr[i * 2 + 1] = lagged_y;

    ylag_xlag_ptr[i * 2] = lagged_y;
    ylag_xlag_ptr[i * 2 + 1] = lagged_x;

    ylag_ptr[i] = lagged_y;
    yt_ptr[i] = current_y;
    xt1_ptr[i] = lagged_x;
  }

  // Create the multivariate array vector
  std::vector<py::array_t<int64_t>> multivar_arrays = {yt, ylag, xt1};

  // Calculate entropies
  double h_y_ylag =
      discrete_joint_entropy(y_ylag, {n_classes[0], n_classes[0]});
  double h_xy_lag =
      discrete_joint_entropy(ylag_xlag, {n_classes[0], n_classes[1]});
  double h_y_lag = discrete_entropy(ylag, n_classes[0]);
  double h_y_ylag_xlag = discrete_multivar_joint_entropy(
      multivar_arrays, {n_classes[0], n_classes[0], n_classes[1]});

  return h_y_ylag + h_xy_lag - h_y_ylag_xlag - h_y_lag;
}

double
discrete_log_normalized_transfer_entropy(const py::array_t<int64_t> &data,
                                         const std::vector<int> &n_classes,
                                         int lag) {

  // Calculate raw transfer entropy
  double te = discrete_transfer_entropy(data, n_classes, lag);

  // Normalize by log of number of classes
  // Use the source variable's n_classes (index 1 since it's [Y,X])
  return te / std::log(static_cast<double>(n_classes[1]));
}

} // namespace fast_entropy
