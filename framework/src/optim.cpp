#include "deeplearn/optim.h"

namespace dl {

SGD::SGD(const std::vector<Tensor *> &parameters, float learning_rate,
         float momentum)
    : _parameters(parameters), _lr(learning_rate), _momentum(momentum) {

  // Initialize velocity buffers if momentum is enabled
  if (_momentum > 0.0f) {
    for (Tensor *param : _parameters) {
      if (param != nullptr) {
        // Create a velocity buffer with the same shape as the parameter
        Tensor velocity(param->shape());
        velocity.zero_();
        _velocity_buffers[param] = velocity;
      }
    }
  }
}

void SGD::step() {
  for (Tensor *param : _parameters) {
    if (param == nullptr || param->grad == nullptr) {
      continue;
    }

    if (_momentum > 0.0f) {
      // SGD with momentum: v = momentum * v + lr * grad
      //                    W = W - v
      auto &velocity = _velocity_buffers[param];

      // velocity = momentum * velocity + lr * grad
      for (size_t i = 0; i < param->numel(); ++i) {
        velocity._data[i] =
            _momentum * velocity._data[i] + _lr * param->grad->_data[i];
        param->_data[i] -= velocity._data[i];
      }
    } else {
      // Standard SGD: W = W - lr * grad
      for (size_t i = 0; i < param->numel(); ++i) {
        param->_data[i] -= _lr * param->grad->_data[i];
      }
    }
  }
}

void SGD::zero_grad() {
  for (Tensor *param : _parameters) {
    if (param != nullptr) {
      param->zero_grad();
    }
  }
}

} // namespace dl
