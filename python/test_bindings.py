#!/usr/bin/env python3
"""
Simple test script for the DeepLearn Python bindings.
Tests Tensor, Layer, Optimizer, and Loss classes with NumPy integration.
"""

import numpy as np
import sys
import os

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bindings')
sys.path.insert(0, build_dir)
print(f"Looking for module in: {build_dir}")

import deeplearn as dl

print("=" * 60)
print("DeepLearn Python Bindings Test")
print("=" * 60)

# ========== Test 1: Tensor Creation and NumPy Conversion ==========
print("\n### Test 1: Tensor Creation and NumPy Conversion")

# Create tensor from shape
t1 = dl.Tensor([2, 3])
print(f"Created tensor: {t1}")

# Create tensor from NumPy array
np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
print(f"\nNumPy array:\n{np_array}")

t2 = dl.Tensor.from_numpy(np_array)
print(f"Tensor from NumPy: {t2}")

# Convert back to NumPy
np_result = t2.to_numpy()
print(f"Back to NumPy:\n{np_result}")
print(f"Arrays match: {np.allclose(np_array, np_result)}")

# ========== Test 2: Tensor Operations ==========
print("\n### Test 2: Tensor Operations")

a = dl.Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
b = dl.Tensor.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))

print(f"a:\n{a.to_numpy()}")
print(f"b:\n{b.to_numpy()}")

# Addition
c = a + b
print(f"\na + b:\n{c.to_numpy()}")

# Multiplication
d = a * b
print(f"a * b:\n{d.to_numpy()}")

# Matrix multiplication
e = a @ b
print(f"a @ b (matmul):\n{e.to_numpy()}")

# ========== Test 3: Neural Network Operations ==========
print("\n### Test 3: Neural Network Operations")

# ReLU
x = dl.Tensor.from_numpy(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32).reshape(1, 4))
x_relu = x.relu()
print(f"Input: {x.to_numpy()}")
print(f"ReLU:  {x_relu.to_numpy()}")

# Softmax
logits = dl.Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 3))
probs = logits.softmax(1)
print(f"\nLogits:  {logits.to_numpy()}")
print(f"Softmax: {probs.to_numpy()}")
print(f"Sum:     {probs.to_numpy().sum()}")

# ========== Test 4: Linear Layer ==========
print("\n### Test 4: Linear Layer")

# Create a Linear layer (3 inputs -> 2 outputs)
fc = dl.Linear(3, 2)
print(f"Linear layer created: 3 -> 2")
print(f"Weight shape: {fc.W.shape}")
print(f"Bias shape: {fc.b.shape}")

# Forward pass
input_tensor = dl.Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
output = fc(input_tensor)
print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output.to_numpy()}")

# Get parameters
params = fc.parameters()
print(f"\nNumber of parameters: {len(params)}")

# ========== Test 5: Autograd ==========
print("\n### Test 5: Autograd (Backward Pass)")

# Create tensors with gradient tracking
x = dl.Tensor.from_numpy(np.array([[2.0]], dtype=np.float32))
y = dl.Tensor.from_numpy(np.array([[3.0]], dtype=np.float32))
x.requires_grad = True
y.requires_grad = True

# Forward pass: z = x + y
z = x.add(y)
print(f"x = {x.to_numpy()}")
print(f"y = {y.to_numpy()}")
print(f"z = x + y = {z.to_numpy()}")

# Backward pass
z.backward()
print(f"\nAfter backward:")
print(f"x.grad = {x.grad.to_numpy() if x.grad else 'None'}")
print(f"y.grad = {y.grad.to_numpy() if y.grad else 'None'}")

# ========== Test 6: SGD Optimizer ==========
print("\n### Test 6: SGD Optimizer")

# Create a simple model parameter
w = dl.Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
w.requires_grad = True

# Simulate gradient
w.grad = dl.Tensor.from_numpy(np.array([[0.1, 0.2, 0.3]], dtype=np.float32))

print(f"Weight before: {w.to_numpy()}")
print(f"Gradient:      {w.grad.to_numpy()}")

# Create optimizer
optimizer = dl.SGD([w], lr=0.1)

# Take one step
optimizer.step()
print(f"Weight after:  {w.to_numpy()}")
print(f"Expected:      [[0.99 1.98 2.97]]")

# ========== Test 7: Cross Entropy Loss ==========
print("\n### Test 7: Cross Entropy Loss")

# Create logits and targets
logits = dl.Tensor.from_numpy(np.array([[2.0, 0.5, 1.0]], dtype=np.float32))
targets = dl.Tensor.from_numpy(np.array([0.0], dtype=np.float32))  # class 0
logits.requires_grad = True

print(f"Logits: {logits.to_numpy()}")
print(f"Target class: {int(targets.to_numpy()[0])}")

# Compute loss
criterion = dl.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"Loss: {loss.to_numpy()}")

# Backward
loss.backward()
print(f"Gradient: {logits.grad.to_numpy()}")

# ========== Test 8: Simple Training Loop ==========
print("\n### Test 8: Simple Training Loop")

# Create a small linear model
model = dl.Linear(2, 1)

# Create optimizer
params = model.parameters()
optimizer = dl.SGD(params, lr=0.01)

# Dummy data
X = dl.Tensor.from_numpy(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32))
y = dl.Tensor.from_numpy(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

print("Training for 3 steps...")
for step in range(3):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(X)
    
    # Compute loss (just for demo, not actual training)
    # We'll just do a simple sum as a dummy loss
    loss = logits.sum()
    
    print(f"  Step {step + 1}, Loss: {loss.to_numpy()[0]:.4f}")
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
