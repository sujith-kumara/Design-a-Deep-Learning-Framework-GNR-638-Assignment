#!/usr/bin/env python3
"""
Evaluation script for the deep learning framework.
Usage: python3 evaluate.py --dataset <path> --weights <path>
"""
import argparse
import os
import sys
import time

# Add build directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl
from dataloader import DataLoader
from model import SimpleCNN


def evaluate(args):
    dataset_path = args.dataset
    weights_path = args.weights
    batch_size = args.batch_size
    test_split = args.test_split
    seed = args.seed
    
    print(f"--- CNN Evaluation on {dataset_path} ---")
    print(f"Weights: {weights_path}, Test Split: {test_split}, Seed: {seed}")
    
    # Load dataset with same parameters as training
    loader = DataLoader(dataset_path, batch_size=batch_size, shuffle=False, 
                      test_split=test_split, seed=seed)
    loader.set_mode('test')
    
    if len(loader) == 0:
        print("Warning: Test set is empty. Evaluate on full dataset instead.")
        loader = DataLoader(dataset_path, batch_size=batch_size, shuffle=False)
    
    # Device selection
    device = dl.Device.GPU
    print(f"Device selected: {device}")
    
    # Initialize Model
    model = SimpleCNN(num_classes=len(loader.classes))
    model.to(device)
    model.print_stats()
    
    # Load weights
    print(f"\nLoading weights from {weights_path}...")
    model.load_weights(weights_path)
    print("✓ Weights loaded successfully")
    
    # Initialize loss
    criterion = dl.CrossEntropyLoss()
    
    # Evaluation loop
    print("\nEvaluating...")
    eval_start = time.time()
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    for images, labels, load_time in loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass (no gradient needed)
        logits = model(images)
        
        # Calculate loss
        loss = criterion(logits, labels)
        total_loss += loss.to_numpy().item()
        
        # Calculate accuracy
        predictions = logits.to_numpy().argmax(axis=1)
        labels_np = labels.to_numpy()
        correct += (predictions == labels_np).sum()
        total += len(labels_np)
        num_batches += 1
    
    eval_time = time.time() - eval_start
    
    # Report results
    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    
    print("\n========================================")
    print("          Evaluation Results")
    print("========================================")
    print(f"Total Samples:    {total}")
    print(f"Average Loss:     {avg_loss:.4f}")
    print(f"Accuracy:         {accuracy:.2f}%")
    print(f"Evaluation Time:  {eval_time:.2f} seconds")
    print(f"Throughput:       {total/eval_time:.2f} samples/sec")
    print("========================================")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_samples': total,
        'eval_time': eval_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained CNN model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to test dataset directory')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to saved weights file (.npz)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for testing (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training)')
    
    args = parser.parse_args()
    evaluate(args)
