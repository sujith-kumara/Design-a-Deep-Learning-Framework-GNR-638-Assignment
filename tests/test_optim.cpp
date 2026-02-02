#include "deeplearn/optim.h"
#include "deeplearn/tensor.h"
#include <iostream>
#include <vector>

using namespace dl;

int main() {
  std::cout << "=== Testing SGD Optimizer ===" << std::endl;

  // Create some test parameters (weights)
  Tensor w1({2, 3});
  Tensor w2({3, 1});

  // Initialize with some values
  w1({0, 0}) = 1.0f;
  w1({0, 1}) = 2.0f;
  w1({0, 2}) = 3.0f;
  w1({1, 0}) = 4.0f;
  w1({1, 1}) = 5.0f;
  w1({1, 2}) = 6.0f;

  w2({0, 0}) = 0.5f;
  w2({1, 0}) = 1.0f;
  w2({2, 0}) = 1.5f;

  // Mark as requiring gradients
  w1.requires_grad = true;
  w2.requires_grad = true;

  std::cout << "\n--- Initial Weights ---" << std::endl;
  std::cout << "w1:" << std::endl;
  w1.print();
  std::cout << "w2:" << std::endl;
  w2.print();

  // Simulate some gradients (normally computed by backward pass)
  w1.grad = new Tensor(w1.shape());
  w1.grad->operator()({0, 0}) = 0.1f;
  w1.grad->operator()({0, 1}) = 0.2f;
  w1.grad->operator()({0, 2}) = 0.3f;
  w1.grad->operator()({1, 0}) = 0.1f;
  w1.grad->operator()({1, 1}) = 0.2f;
  w1.grad->operator()({1, 2}) = 0.3f;

  w2.grad = new Tensor(w2.shape());
  w2.grad->operator()({0, 0}) = 0.05f;
  w2.grad->operator()({1, 0}) = 0.1f;
  w2.grad->operator()({2, 0}) = 0.15f;

  std::cout << "\n--- Simulated Gradients ---" << std::endl;
  std::cout << "w1.grad:" << std::endl;
  w1.grad->print();
  std::cout << "w2.grad:" << std::endl;
  w2.grad->print();

  // Test 1: Basic SGD without momentum
  std::cout << "\n=== Test 1: SGD without momentum (lr=0.1) ===" << std::endl;
  {
    std::vector<Tensor *> params = {&w1, &w2};
    SGD optimizer(params, 0.1f); // learning rate = 0.1

    std::cout << "Performing one step..." << std::endl;
    optimizer.step();

    std::cout << "Updated weights:" << std::endl;
    std::cout << "w1 (should be: original - 0.1 * grad):" << std::endl;
    w1.print();
    std::cout << "w2 (should be: original - 0.1 * grad):" << std::endl;
    w2.print();
  }

  // Reset weights and gradients for next test
  w1({0, 0}) = 1.0f;
  w1({0, 1}) = 2.0f;
  w1({0, 2}) = 3.0f;
  w1({1, 0}) = 4.0f;
  w1({1, 1}) = 5.0f;
  w1({1, 2}) = 6.0f;

  w2({0, 0}) = 0.5f;
  w2({1, 0}) = 1.0f;
  w2({2, 0}) = 1.5f;

  delete w1.grad;
  delete w2.grad;

  w1.grad = new Tensor(w1.shape());
  w1.grad->operator()({0, 0}) = 0.1f;
  w1.grad->operator()({0, 1}) = 0.2f;
  w1.grad->operator()({0, 2}) = 0.3f;
  w1.grad->operator()({1, 0}) = 0.1f;
  w1.grad->operator()({1, 1}) = 0.2f;
  w1.grad->operator()({1, 2}) = 0.3f;

  w2.grad = new Tensor(w2.shape());
  w2.grad->operator()({0, 0}) = 0.05f;
  w2.grad->operator()({1, 0}) = 0.1f;
  w2.grad->operator()({2, 0}) = 0.15f;

  // Test 2: SGD with momentum
  std::cout << "\n=== Test 2: SGD with momentum (lr=0.1, momentum=0.9) ==="
            << std::endl;
  {
    std::vector<Tensor *> params = {&w1, &w2};
    SGD optimizer(params, 0.1f, 0.9f); // learning rate = 0.1, momentum = 0.9

    std::cout << "Performing first step..." << std::endl;
    optimizer.step();

    std::cout << "Updated weights after step 1:" << std::endl;
    std::cout << "w1:" << std::endl;
    w1.print();
    std::cout << "w2:" << std::endl;
    w2.print();

    // Simulate second step with same gradients
    std::cout << "\nPerforming second step (with accumulated momentum)..."
              << std::endl;
    optimizer.step();

    std::cout << "Updated weights after step 2:" << std::endl;
    std::cout << "w1:" << std::endl;
    w1.print();
    std::cout << "w2:" << std::endl;
    w2.print();
  }

  // Test 3: zero_grad functionality
  std::cout << "\n=== Test 3: zero_grad functionality ===" << std::endl;
  {
    std::vector<Tensor *> params = {&w1, &w2};
    SGD optimizer(params, 0.1f);

    std::cout << "Before zero_grad:" << std::endl;
    std::cout << "w1.grad exists: " << (w1.grad != nullptr ? "yes" : "no")
              << std::endl;
    std::cout << "w2.grad exists: " << (w2.grad != nullptr ? "yes" : "no")
              << std::endl;

    optimizer.zero_grad();

    std::cout << "After zero_grad:" << std::endl;
    std::cout << "w1.grad exists: " << (w1.grad != nullptr ? "yes" : "no")
              << std::endl;
    std::cout << "w2.grad exists: " << (w2.grad != nullptr ? "yes" : "no")
              << std::endl;
  }

  std::cout << "\n=== All SGD tests completed! ===" << std::endl;

  return 0;
}
