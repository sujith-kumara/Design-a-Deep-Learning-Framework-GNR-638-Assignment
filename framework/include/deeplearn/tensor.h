#pragma once

#include <iostream>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

namespace dl {

class Tensor {
public:
  Tensor();
  explicit Tensor(const std::vector<int> &shape);
  Tensor(const std::vector<int> &shape, const std::vector<float> &data);
  ~Tensor();

  // Helper methods
  const std::vector<int> &size() const { return _shape; }
  size_t numel() const;
  void reshape(const std::vector<int> &new_shape);
  void zero_();
  void print() const;
  void backward();
  void zero_grad();

  // Mathematical Operations
  Tensor add(const Tensor &other) const;
  Tensor sub(const Tensor &other) const;
  Tensor mul(const Tensor &other) const;
  Tensor matmul(const Tensor &other) const;
  Tensor transpose() const; // Added
  Tensor sum() const;       // Added

  // Neural Network Operations
  Tensor conv2d(const Tensor &kernel, int stride = 1, int padding = 0) const;
  Tensor maxpool2d(int kernel_size, int stride = 1) const;

  const std::vector<int> &shape() const { return _shape; } // Added

  // Indexing helpers
  float &operator()(const std::vector<int> &indices);
  float operator()(const std::vector<int> &indices) const;

  // Autograd members
  bool requires_grad{false};
  Tensor *grad{nullptr};
  std::vector<Tensor *> parents;
  std::string op_type{""};
  int stride{1};
  int padding{0};
  int kernel_size{0};

private:
  std::vector<float> _data;
  std::vector<int> _shape;

  size_t get_flat_index(const std::vector<int> &indices) const;
};

} // namespace dl
