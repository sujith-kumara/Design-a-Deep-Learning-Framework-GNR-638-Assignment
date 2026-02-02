#pragma once

#include "tensor.h"
#include <unordered_map>
#include <vector>

namespace dl {

// Base optimizer class
class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step() = 0;
  virtual void zero_grad() = 0;
};

// SGD optimizer with optional momentum
class SGD : public Optimizer {
public:
  // Constructor: learning_rate is required, momentum is optional (default 0.0)
  SGD(const std::vector<Tensor *> &parameters, float learning_rate,
      float momentum = 0.0f);

  // Perform one optimization step: W -= lr * W.grad (with momentum if enabled)
  void step() override;

  // Zero out all parameter gradients
  void zero_grad() override;

private:
  std::vector<Tensor *> _parameters;
  float _lr;
  float _momentum;

  // Momentum buffer: stores velocity for each parameter
  std::unordered_map<Tensor *, Tensor> _velocity_buffers;
};

} // namespace dl
