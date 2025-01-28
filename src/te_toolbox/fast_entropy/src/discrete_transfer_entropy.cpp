#include "discrete_transfer_entropy.h"
#include "discrete_entropy.h"
#include "discrete_joint_entropy.h"
#include "discrete_multivar_entropy.h"
#include <stdexcept>
#include <vector>

namespace fast_entropy {

double discrete_transfer_entropy(const py::array_t<int64_t> &data,
                                 const std::vector<int> &n_classes, int lag) {

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

  // Prepare data arrays
  std::vector<int64_t> y_ylag_data(n_current * 2);
  std::vector<int64_t> ylag_xlag_data(n_current * 2);
  std::vector<int64_t> yt1_data(n_current);
  std::vector<int64_t> yt_data(n_current);
  std::vector<int64_t> xt1_data(n_current);

  // Fill arrays in one pass
  for (size_t i = 0; i < n_current; ++i) {
    const int64_t current_y = ptr[(i + lag) * 2]; // Y_t
    const int64_t lagged_y = ptr[i * 2];          // Y_{t-1}
    const int64_t lagged_x = ptr[i * 2 + 1];      // X_{t-1}

    y_ylag_data[i * 2] = current_y;
    y_ylag_data[i * 2 + 1] = lagged_y;

    ylag_xlag_data[i * 2] = lagged_y;
    ylag_xlag_data[i * 2 + 1] = lagged_x;

    yt_data[i] = current_y;
    yt1_data[i] = lagged_y;
    xt1_data[i] = lagged_x;
  }

  // Create arrays for joint entropy
  std::vector<ssize_t> shape = {static_cast<ssize_t>(n_current), 2};
  py::array_t<int64_t> y_ylag(shape);
  py::array_t<int64_t> ylag_xlag(shape);

  auto y_ylag_buf = y_ylag.request();
  auto ylag_xlag_buf = ylag_xlag.request();

  std::memcpy(y_ylag_buf.ptr, y_ylag_data.data(),
              y_ylag_data.size() * sizeof(int64_t));
  std::memcpy(ylag_xlag_buf.ptr, ylag_xlag_data.data(),
              ylag_xlag_data.size() * sizeof(int64_t));

  // Create arrays for single variable entropy
  py::array_t<int64_t> yt1_arr({static_cast<ssize_t>(n_current)});
  auto yt1_buf = yt1_arr.request();
  std::memcpy(yt1_buf.ptr, yt1_data.data(), yt1_data.size() * sizeof(int64_t));

  // Create arrays for multivariate entropy
  py::array_t<int64_t> yt_arr({static_cast<ssize_t>(n_current)});
  py::array_t<int64_t> xt1_arr({static_cast<ssize_t>(n_current)});

  auto yt_buf = yt_arr.request();
  auto xt1_buf = xt1_arr.request();

  std::memcpy(yt_buf.ptr, yt_data.data(), yt_data.size() * sizeof(int64_t));
  std::memcpy(xt1_buf.ptr, xt1_data.data(), xt1_data.size() * sizeof(int64_t));

  std::vector<py::array_t<int64_t>> multivar_arrays = {yt_arr, yt1_arr,
                                                       xt1_arr};

  // Calculate entropies
  double h_y_ylag =
      discrete_joint_entropy(y_ylag, {n_classes[0], n_classes[0]});
  double h_xy_lag =
      discrete_joint_entropy(ylag_xlag, {n_classes[0], n_classes[1]});
  double h_y_lag = discrete_entropy(yt1_arr, n_classes[0]);
  double h_y_ylag_xlag = discrete_multivar_joint_entropy(
      multivar_arrays, {n_classes[0], n_classes[0], n_classes[1]});

  return h_y_ylag + h_xy_lag - h_y_ylag_xlag - h_y_lag;
}

} // namespace fast_entropy
