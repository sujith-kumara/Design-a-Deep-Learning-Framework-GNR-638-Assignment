import os
import time
import sys
import numpy as np
from dataloader import DataLoader
from model import SimpleCNN

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl

def train():
    # Configuration
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1')
    batch_size = 64
    epochs = 3
    learning_rate = 0.01
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_weights.npz')

    print(f"--- CNN Training on {data_dir} ---")
    
    # Initialize DataLoader
    try:
        loader = DataLoader(data_dir, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Initialize Model
    model = SimpleCNN(num_classes=len(loader.classes))
    
    # Initialize Optimizer
    params = model.parameters()
    optimizer = dl.SGD(params, lr=learning_rate)
    
    # Loss function
    criterion = dl.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start = time.time()
        
        total_loss = 0.0
        correct = 0
        total = 0
        batch_idx = 0
        
        for images, labels, load_time in loader:
            batch_idx += 1
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Metrics
            total_loss += loss.to_numpy()[0]
            
            # Calculation of accuracy (argmax on logits)
            logits_np = logits.to_numpy()
            labels_np = labels.to_numpy().astype(int)
            predictions = np.argmax(logits_np, axis=1)
            correct += np.sum(predictions == labels_np)
            total += labels_np.shape[0]
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.to_numpy()[0]:.4f}, Acc: {100.0 * correct / total:.2f}%", flush=True)

        epoch_end = time.time()
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        print(f"\n  Final Metrics - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%", flush=True)
        print(f"  Epoch Time: {epoch_end - epoch_start:.2f} seconds")

    # Save weights
    model.save_weights(weights_path)

if __name__ == "__main__":
    train()
