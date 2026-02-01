#pragma once

#include <iostream>
#include <vector>

namespace dl {

class Tensor {
public:
  Tensor();
  explicit Tensor(const std::vector<int> &shape);
  ~Tensor();

  // Helper methods
  const std::vector<int> &size() const { return _shape; }
  size_t numel() const;
  void reshape(const std::vector<int> &new_shape);
  void zero_();
  void print() const;

  // Autograd-ready members (no logic yet)
  bool requires_grad{false};
  Tensor *grad{nullptr};

private:
  std::vector<float> _data;
  std::vector<int> _shape;
};

} // namespace dl
