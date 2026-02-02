#include "deeplearn/layers.h"
#include "deeplearn/loss.h"
#include "deeplearn/optim.h"
#include <iostream>
#include <vector>

using namespace dl;

int main() {
  std::cout << "=== SGD Optimizer with Neural Network Example ===" << std::endl;

  // Create a simple Linear layer (3 features -> 2 outputs)
  Linear fc(3, 2);

  std::cout << "\n--- Initial Layer Weights ---" << std::endl;
  std::cout << "W:" << std::endl;
  fc.W.print();
  std::cout << "b:" << std::endl;
  fc.b.print();

  // Collect all trainable parameters
  std::vector<Tensor *> parameters = {&fc.W, &fc.b};

  // Create SGD optimizer with lr=0.01 and momentum=0.9
  SGD optimizer(parameters, 0.01f, 0.9f);

  // Create dummy input and target for one training step
  Tensor input({1, 3});
  input({0, 0}) = 1.0f;
  input({0, 1}) = 2.0f;
  input({0, 2}) = 3.0f;

  Tensor target({1}, {0.0f}); // Target class 0

  std::cout << "\n=== Training Loop (5 steps) ===" << std::endl;
  for (int step = 0; step < 5; ++step) {
    std::cout << "\n--- Step " << (step + 1) << " ---" << std::endl;

    // Zero gradients from previous step
    optimizer.zero_grad();

    // Forward pass
    Tensor output = fc.forward(input);

    // Compute loss
    CrossEntropyLoss criterion;
    Tensor loss = criterion(output, target);

    std::cout << "Loss: ";
    loss.print();

    // Backward pass
    loss.backward();

    // Update parameters
    optimizer.step();

    // Print updated weights (only on first and last step)
    if (step == 0 || step == 4) {
      std::cout << "W (after update):" << std::endl;
      fc.W.print();
    }
  }

  std::cout << "\n=== Training completed! ===" << std::endl;
  std::cout << "The weights have been updated using SGD with momentum."
            << std::endl;

  return 0;
}
