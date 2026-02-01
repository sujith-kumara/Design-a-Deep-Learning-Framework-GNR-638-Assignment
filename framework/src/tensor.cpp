#include "deeplearn/tensor.h"
#include <iomanip>
#include <iostream> // Added for std::cout, std::endl
#include <numeric>

namespace dl {

Tensor::Tensor() : grad(nullptr), requires_grad(false) {}

Tensor::Tensor(const std::vector<int> &shape)
    : _shape(shape), grad(nullptr), requires_grad(false) {
  size_t total_size = numel();
  _data.assign(total_size, 0.0f);
}

Tensor::~Tensor() {
  // Note: grad ownership should be handled carefully later in autograd
}

size_t Tensor::numel() const {
  if (_shape.empty())
    return 0;
  size_t total = 1;
  for (int s : _shape)
    total *= s;
  return total;
}

void Tensor::reshape(const std::vector<int> &new_shape) {
  size_t new_total = 1;
  for (int s : new_shape)
    new_total *= s;

  if (new_total != numel()) {
    throw std::runtime_error(
        "Reshape failed: total number of elements must remain the same");
  }
  _shape = new_shape;
}

void Tensor::zero_() { std::fill(_data.begin(), _data.end(), 0.0f); }

void Tensor::print() const {
  std::cout << "Tensor(shape=[";
  for (size_t i = 0; i < _shape.size(); ++i) {
    std::cout << _shape[i] << (i == _shape.size() - 1 ? "" : ", ");
  }
  std::cout << "], data=[";

  size_t n = numel();
  for (size_t i = 0; i < n; ++i) {
    std::cout << std::fixed << std::setprecision(2) << _data[i]
              << (i == n - 1 ? "" : ", ");
  }
  std::cout << "])" << std::endl;
}

} // namespace dl
