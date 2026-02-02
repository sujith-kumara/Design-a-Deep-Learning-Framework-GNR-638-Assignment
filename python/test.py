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

def evaluate():
    # Configuration
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1') # Evaluation usually on test set
    batch_size = 64
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_weights.npz')

    print(f"--- CNN Evaluation on {data_dir} ---")
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights file {weights_path} not found. Please train first.")
        return

    # Initialize DataLoader
    try:
        loader = DataLoader(data_dir, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Initialize Model and Load Weights
    model = SimpleCNN(num_classes=len(loader.classes))
    model.load_weights(weights_path)
    
    correct = 0
    total = 0
    inference_times = []
    
    print("Evaluating...")
    for images, labels, load_time in loader:
        # Forward pass (no grad tracking needed ideally, but our backend might track anyway)
        # Note: We don't have a 'no_grad' context yet in the backend, but we can manually disable it if needed.
        # However, for inference-only, we just won't call backward.
        
        start_inf = time.time()
        logits = model(images)
        end_inf = time.time()
        
        inference_times.append(end_inf - start_inf)
        
        # Metrics
        logits_np = logits.to_numpy()
        labels_np = labels.to_numpy().astype(int)
        predictions = np.argmax(logits_np, axis=1)
        correct += np.sum(predictions == labels_np)
        total += labels_np.shape[0]

    accuracy = 100.0 * correct / total
    avg_inf_time = (sum(inference_times) / total) * 1000 # ms per image
    
    print("\n--- Evaluation Results ---")
    print(f"Total Samples: {total}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print(f"Avg Inf Time:  {avg_inf_time:.4f} ms per image")

if __name__ == "__main__":
    evaluate()
