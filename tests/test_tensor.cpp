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
  dl::Tensor pool_out = input.maxpool2d(2, 1);
  assert(pool_out.size()[2] == 2);
  assert(pool_out({0, 0, 0, 0}) == 1.0f);
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

  // Simple scalar chain: (x * y) + z
  dl::Tensor x({1});
  x({0}) = 2.0f;
  x.requires_grad = true;
  dl::Tensor y({1});
  y({0}) = 3.0f;
  y.requires_grad = true;
  dl::Tensor z({1});
  z({0}) = 4.0f;
  z.requires_grad = true;

  dl::Tensor xy = x.mul(y); // Keep intermediate alive
  dl::Tensor res = xy.add(z);
  res.backward();

  // dres/dx = y = 3
  assert(x.grad->operator()({0}) == 3.0f);
  // dres/dy = x = 2
  assert(y.grad->operator()({0}) == 2.0f);
  // dres/dz = 1
  assert(z.grad->operator()({0}) == 1.0f);
  std::cout << "Backward test (mul + add chain) passed." << std::endl;

  std::cout << "All basic Tensor tests passed!" << std::endl;
  return 0;
}
