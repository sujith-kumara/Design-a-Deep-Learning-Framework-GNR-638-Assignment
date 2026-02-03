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

def train(args):
    # Configuration
    data_dir = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    weights_path = args.save_path

    print(f"--- CNN Training on {data_dir} ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    
    # Initialize DataLoader
    try:
        loader = DataLoader(data_dir, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize Model
    model = SimpleCNN(num_classes=len(loader.classes))
    model.print_stats()
    
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

            # Print every few batches
            if batch_idx % 5 == 0 or batch_idx == len(loader):
                print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.to_numpy()[0]:.4f}, Acc: {100.0 * correct / total:.2f}%", flush=True)

        epoch_end = time.time()
        # Avoid division by zero if loader is empty (though specific exception above handles empty/bad dirs usually)
        if len(loader) > 0:
            avg_loss = total_loss / len(loader)
        else:
            avg_loss = 0
            
        if total > 0:
            accuracy = 100.0 * correct / total
        else:
            accuracy = 0.0
            
        print(f"\n  Final Metrics - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%", flush=True)
        print(f"  Epoch Time: {epoch_end - epoch_start:.2f} seconds")

    # Save weights
    print(f"Saving weights to {weights_path}")
    model.save_weights(weights_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN on a custom dataset')
    parser.add_argument('--dataset', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1'),
                        help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_weights.npz'),
                        help='Path to save weights')
    
    args = parser.parse_args()
    train(args)
