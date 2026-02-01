#include "tensor.h"
#include <vector>

namespace dl {

// Placeholder for neural network layers
class Layer {
public:
  virtual ~Layer() = default;
};

class Conv2D : public Layer {
public:
  Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1,
         int padding = 0);
  Tensor forward(const Tensor &input);

  Tensor W;
  Tensor b;
  int stride;
  int padding;

private:
  Tensor _conv_out; // Store intermediate to keep it alive for autograd
};

} // namespace dl
