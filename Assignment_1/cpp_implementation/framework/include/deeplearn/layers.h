#include "tensor.h"
#include <vector>

namespace dl {

// Neural network layer abstractions
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

class Linear : public Layer {
public:
  Linear(int in_features, int out_features);
  Tensor forward(const Tensor &input);

  Tensor W;
  Tensor b;

private:
  Tensor _w_transpose;
  Tensor _matmul_out;
};

} // namespace dl
