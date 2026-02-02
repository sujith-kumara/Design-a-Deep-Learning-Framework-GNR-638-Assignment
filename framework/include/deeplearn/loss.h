#pragma once
#include "tensor.h"

namespace dl {

// Placeholder for loss functions
class Loss {
public:
  virtual ~Loss() = default;
};

class CrossEntropyLoss : public Loss {
public:
  Tensor operator()(const Tensor &input, const Tensor &target) {
    return input.cross_entropy(target);
  }
};

} // namespace dl
