#pragma once

#include <iostream>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

namespace dl {

enum class Device { CPU, GPU };

class Tensor {
public:
  Tensor();
  explicit Tensor(const std::vector<int> &shape, Device device = Device::CPU);
  Tensor(const std::vector<int> &shape, const std::vector<float> &data,
         Device device = Device::CPU);
  ~Tensor();

  // Device management
  Tensor to(Device device);
  Device device() const { return _device; }

  // Copy/Move semantics
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

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
  Tensor relu() const;                              // Added
  Tensor softmax(int dim = -1) const;               // Added
  Tensor cross_entropy(const Tensor &target) const; // Added
  Tensor conv2d(const Tensor &kernel, int stride = 1, int padding = 0) const;
  Tensor maxpool2d(int kernel_size, int stride = 1) const;

  const std::vector<int> &shape() const { return _shape; } // Added

  // Data access (for Python bindings and serialization)
  const std::vector<float> &data() const { return _data; }
  std::vector<float> &data() { return _data; }

  // Indexing helpers
  float &operator()(const std::vector<int> &indices);
  float operator()(const std::vector<int> &indices) const;

  // Faster access for internal loops
  float &operator[](size_t index) { return _data[index]; }
  const float &operator[](size_t index) const { return _data[index]; }
  float *data_ptr() { return _data.data(); }
  const float *data_ptr() const { return _data.data(); }

  // Autograd members
  bool requires_grad{false};
  Tensor *grad{nullptr};
  std::vector<Tensor *> parents;
  std::string op_type{""};
  int stride{1};
  int padding{0};
  int kernel_size{0};

private:
  friend class SGD; // Allow SGD to access _data for optimization
  std::vector<float> _data;
  std::vector<int> _shape;
  std::vector<size_t> _max_indices; // Added for maxpool backward
  Device _device{Device::CPU};

  size_t get_flat_index(const std::vector<int> &indices) const;
};

} // namespace dl
