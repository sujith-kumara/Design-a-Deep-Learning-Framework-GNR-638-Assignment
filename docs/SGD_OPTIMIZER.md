# SGD Optimizer Implementation

## Overview
This implementation provides a **Stochastic Gradient Descent (SGD)** optimizer with optional momentum support for training neural networks in the DeepLearn framework.

## Features
- ✅ Basic SGD with configurable learning rate
- ✅ Momentum support for accelerated convergence
- ✅ Batch parameter updates
- ✅ Gradient zeroing functionality

## Usage

### Basic SGD (without momentum)
```cpp
#include "deeplearn/optim.h"
#include "deeplearn/layers.h"

// Create a neural network layer
dl::Linear layer(10, 5);

// Collect all trainable parameters
std::vector<dl::Tensor*> parameters = {&layer.W, &layer.b};

// Create SGD optimizer with learning rate = 0.01
dl::SGD optimizer(parameters, 0.01f);

// Training loop
for (int epoch = 0; epoch < 100; ++epoch) {
    // Zero gradients
    optimizer.zero_grad();
    
    // Forward pass
    dl::Tensor output = layer.forward(input);
    
    // Compute loss
    dl::Tensor loss = /* your loss computation */;
    
    // Backward pass
    loss.backward();
    
    // Update parameters using: W = W - lr * grad
    optimizer.step();
}
```

### SGD with Momentum
```cpp
// Create SGD optimizer with learning rate = 0.01 and momentum = 0.9
dl::SGD optimizer(parameters, 0.01f, 0.9f);

// The update rule becomes:
// v = momentum * v + lr * grad
// W = W - v
```

## API Reference

### Constructor
```cpp
SGD(const std::vector<Tensor*>& parameters, 
    float learning_rate, 
    float momentum = 0.0f)
```

**Parameters:**
- `parameters`: Vector of pointers to all trainable tensors (weights, biases)
- `learning_rate`: Learning rate (step size) for parameter updates
- `momentum`: Momentum coefficient (default: 0.0, range: [0, 1])

### Methods

#### `void step()`
Performs one optimization step. Updates all parameters based on their gradients.

**Update rules:**
- Without momentum: `W = W - lr * grad`
- With momentum: `v = momentum * v + lr * grad; W = W - v`

#### `void zero_grad()`
Zeros out all parameter gradients by calling `zero_grad()` on each tensor.

## Mathematical Background

### Basic SGD
For each parameter **W** with gradient **∇W**:

```
W ← W - η * ∇W
```

where **η** is the learning rate.

### SGD with Momentum
Momentum helps accelerate SGD in relevant directions and dampens oscillations:

```
v_t ← β * v_{t-1} + η * ∇W
W ← W - v_t
```

where:
- **v** is the velocity (momentum buffer)
- **β** is the momentum coefficient (typically 0.9)
- **η** is the learning rate

## Testing

Run the unit tests:
```bash
./build/tests/test_optim
```

Run the training example:
```bash
./build/tests/example_sgd_training
```

## Implementation Details

- **Memory efficient**: Momentum buffers are only allocated when momentum > 0
- **Type safe**: Uses `std::unordered_map` to associate velocity buffers with parameters
- **Friend access**: The `SGD` class is declared as a friend of `Tensor` to efficiently access `_data`

## Example Output

```
=== Training Loop (5 steps) ===

--- Step 1 ---
Loss: Tensor(shape=[1], data=[0.61])

--- Step 2 ---
Loss: Tensor(shape=[1], data=[0.55])

--- Step 3 ---
Loss: Tensor(shape=[1], data=[0.45])

--- Step 4 ---
Loss: Tensor(shape=[1], data=[0.34])

--- Step 5 ---
Loss: Tensor(shape=[1], data=[0.25])
```

The loss consistently decreases, demonstrating successful optimization!

## Files

- **Header**: `framework/include/deeplearn/optim.h`
- **Implementation**: `framework/src/optim.cpp`
- **Tests**: `tests/test_optim.cpp`
- **Example**: `tests/example_sgd_training.cpp`
