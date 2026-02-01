#include "deeplearn/tensor.h"
#include <cassert>
#include <iostream>

int main() {
  // 1. Initialization
  dl::Tensor t1({2, 3});
  assert(t1.numel() == 6);
  std::cout << "Initialization test passed. Numel: " << t1.numel() << std::endl;

  // 2. Element-wise ops
  dl::Tensor a({2, 2});
  dl::Tensor b({2, 2});
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      a({i, j}) = i + j;
      b({i, j}) = 1.0f;
    }

  dl::Tensor c = a.add(b);
  assert(c({0, 0}) == 1.0f);
  assert(c({1, 1}) == 3.0f);
  std::cout << "Add test passed." << std::endl;

  dl::Tensor d = a.matmul(b); // [[0,1],[1,2]] * [[1,1],[1,1]] = [[1,1],[3,3]]
  assert(d({0, 0}) == 1.0f);
  assert(d({1, 0}) == 3.0f);
  std::cout << "Matmul test passed." << std::endl;

  // 3. Conv2d
  // Input 1x1x3x3, Kernel 1x1x2x2
  dl::Tensor input({1, 1, 3, 3});
  for (int i = 0; i < 9; ++i)
    input.zero_(); // Just zero it for now or fill with ones
  input.zero_();   // filler
  // Fill with values
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      input({0, 0, i, j}) = 1.0f;

  dl::Tensor kernel({1, 1, 2, 2});
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      kernel({0, 0, i, j}) = 1.0f;

  dl::Tensor output =
      input.conv2d(kernel, 1, 0); // Output should be 2x2 with values 4.0
  assert(output.size()[2] == 2);
  assert(output({0, 0, 0, 0}) == 4.0f);
  std::cout << "Conv2d test passed." << std::endl;

  // 4. Maxpool2d
  dl::Tensor input_pool({1, 1, 4, 4});
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      input_pool({0, 0, i, j}) = (float)(i * 4 + j);
  // Input:
  // 0  1  2  3
  // 4  5  6  7
  // 8  9  10 11
  // 12 13 14 15

  input_pool.requires_grad = true;
  dl::Tensor pool_out = input_pool.maxpool2d(2, 2);
  // Pool out (2x2):
  // 5  7
  // 13 15
  assert(pool_out.size()[2] == 2);
  assert(pool_out({0, 0, 0, 0}) == 5.0f);
  assert(pool_out({0, 0, 1, 1}) == 15.0f);

  dl::Tensor pool_loss = pool_out.sum();
  pool_loss.backward();

  // Gradients should be 1.0 only at positions [0,0,1,1], [0,0,1,3], [0,0,3,1],
  // [0,0,3,3] which correspond to 5, 7, 13, 15
  assert(input_pool.grad->operator()({0, 0, 1, 1}) == 1.0f);
  assert(input_pool.grad->operator()({0, 0, 1, 3}) == 1.0f);
  assert(input_pool.grad->operator()({0, 0, 3, 1}) == 1.0f);
  assert(input_pool.grad->operator()({0, 0, 3, 3}) == 1.0f);
  assert(input_pool.grad->operator()({0, 0, 0, 0}) == 0.0f);

  std::cout << "Maxpool2d test passed." << std::endl;

  // 5. Graph metadata and backward
  dl::Tensor a_s({1});
  a_s({0}) = 1.0f;
  dl::Tensor b_s({1});
  b_s({0}) = 2.0f;
  a_s.requires_grad = true;
  b_s.requires_grad = true;
  dl::Tensor e = a_s.add(b_s);
  assert(e.requires_grad == true);
  assert(e.op_type == "add");

  e.backward();
  assert(a_s.grad != nullptr);
  assert(a_s.grad->operator()({0}) == 1.0f);
  assert(b_s.grad != nullptr);
  assert(b_s.grad->operator()({0}) == 1.0f);
  std::cout << "Backward test (add) passed." << std::endl;

  // 6. Diamond graph (shared node)
  dl::Tensor x_d({1});
  x_d({0}) = 2.0f;
  x_d.requires_grad = true;
  dl::Tensor y_d({1});
  y_d({0}) = 3.0f;
  y_d.requires_grad = true;

  dl::Tensor path1 = x_d.mul(y_d);    // 6
  dl::Tensor path2 = x_d.add(y_d);    // 5
  dl::Tensor loss = path1.add(path2); // 11

  loss.backward();
  // dloss/dx = dpath1/dx + dpath2/dx = y_d + 1 = 3 + 1 = 4
  // dloss/dy = dpath1/dy + dpath2/dy = x_d + 1 = 2 + 1 = 3
  assert(x_d.grad->operator()({0}) == 4.0f);
  assert(y_d.grad->operator()({0}) == 3.0f);
  std::cout << "Diamond graph test passed." << std::endl;

  // 7. ReLU test
  dl::Tensor x_relu({2});
  x_relu({0}) = 1.0f;
  x_relu({1}) = -1.0f;
  x_relu.requires_grad = true;
  dl::Tensor y_relu = x_relu.relu();
  assert(y_relu({0}) == 1.0f);
  assert(y_relu({1}) == 0.0f);

  dl::Tensor loss_relu = y_relu.sum();
  loss_relu.backward();
  assert(x_relu.grad->operator()({0}) == 1.0f);
  assert(x_relu.grad->operator()({1}) == 0.0f);
  std::cout << "ReLU test passed." << std::endl;

  // 8. Softmax test
  dl::Tensor x_sm({2});
  x_sm({0}) = 1.0f;
  x_sm({1}) = 1.0f; // Softmax([1,1]) = [0.5, 0.5]
  x_sm.requires_grad = true;
  dl::Tensor y_sm = x_sm.softmax(0);
  assert(std::abs(y_sm({0}) - 0.5f) < 1e-5);
  assert(std::abs(y_sm({1}) - 0.5f) < 1e-5);

  // Backward check

  // Re-run with sum
  x_sm.zero_grad();
  y_sm = x_sm.softmax(0);
  // dx = y * (dy - sum(y*dy))
  // if dy = [1, 1] (from y.sum())
  // sum(y*dy) = 0.5*1 + 0.5*1 = 1
  // dx_0 = 0.5 * (1 - 1) = 0
  // dx_1 = 0.5 * (1 - 1) = 0
  dl::Tensor loss_sm = y_sm.sum();
  loss_sm.backward();
  assert(std::abs(x_sm.grad->operator()({0})) < 1e-5);
  assert(std::abs(x_sm.grad->operator()({1})) < 1e-5);
  std::cout << "Softmax test passed." << std::endl;

  std::cout << "All basic Tensor tests passed!" << std::endl;
  return 0;
}
