import os
import sys
import numpy as np

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl
from model import SimpleCNN

def debug():
    print("Initializing model...")
    model = SimpleCNN(num_classes=10)
    
    print("Starting loop...")
    for iter in range(20):
        print(f"Iteration {iter+1}")
        
        print("  Creating dummy input...")
        x_np = np.random.randn(64, 3, 32, 32).astype(np.float32)
        x = dl.Tensor.from_numpy(x_np)
        x.requires_grad = True
        
        y_np = np.random.randint(0, 10, size=64).astype(np.float32)
        y = dl.Tensor.from_numpy(y_np)
        
        print("  Forward pass...")
        logits = model(x)
        
        print("  Loss calculation...")
        criterion = dl.CrossEntropyLoss()
        loss = criterion(logits, y)
        
        print("  Backward pass...")
        loss.backward()
        
        print("  Optimizer step...")
        params = model.parameters()
        optimizer = dl.SGD(params, lr=0.01)
        optimizer.step()
        
        # Explicitly zero grad or something?
        # The optimizer doesn't zero grad, we do it manually in train.py
        optimizer.zero_grad()

    print("Loop completed.")

if __name__ == "__main__":
    try:
        debug()
        print("Success!")
    except Exception as e:
        print(f"Caught exception: {e}")
