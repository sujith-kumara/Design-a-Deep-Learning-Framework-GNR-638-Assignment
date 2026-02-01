#include "deeplearn/tensor.h"
#include <cassert>
#include <iostream>

int main() {
  // 1. Initialization
  dl::Tensor t1({2, 3});
  assert(t1.numel() == 6);
  std::cout << "Initialization test passed. Numel: " << t1.numel() << std::endl;

  // 2. Reshape
  t1.reshape({3, 2});
  assert(t1.size()[0] == 3);
  assert(t1.size()[1] == 2);
  std::cout << "Reshape test passed." << std::endl;

  // 3. Zero_
  t1.zero_();
  std::cout << "Zero_ test passed (manual check via print next)." << std::endl;

  // 4. Print
  t1.print();

  std::cout << "All basic Tensor tests passed!" << std::endl;
  return 0;
}
