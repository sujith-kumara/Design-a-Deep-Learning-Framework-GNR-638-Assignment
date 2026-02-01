#include "deeplearn/layers.h"
#include "deeplearn/tensor.h"
#include <cassert>
#include <iostream>
#include <vector>

void test_conv2d_layer() {
  // 1. Forward Pass Shape Test
  dl::Conv2D conv(3, 16, 3, 1, 1);  // in=3, out=16, k=3, s=1, p=1
  dl::Tensor input({1, 3, 32, 32}); // B=1, C=3, H=32, W=32

  dl::Tensor output = conv.forward(input);

  // Output shape should be (1, 16, 32, 32) since s=1, p=1, k=3 keeps size
  assert(output.shape()[0] == 1);
  assert(output.shape()[1] == 16);
  assert(output.shape()[2] == 32);
  assert(output.shape()[3] == 32);
  std::cout << "Conv2D forward shape test passed." << std::endl;

  // 2. Backward Pass Test
  dl::Tensor loss = output.sum(); // Dummy loss
  loss.backward();

  assert(conv.W.grad != nullptr);
  assert(conv.b.grad != nullptr);

  // Bias gradient should be filled (16 values)
  float bias_grad_sum = 0;
  for (int i = 0; i < 16; ++i)
    bias_grad_sum += conv.b.grad->operator()({0, i, 0, 0});
  assert(bias_grad_sum != 0);

  std::cout << "Conv2D backward test passed." << std::endl;
}

int main() {
  test_conv2d_layer();
  return 0;
}
