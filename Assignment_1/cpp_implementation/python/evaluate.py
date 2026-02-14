#!/usr/bin/env python3
import argparse
import os
import sys
import time
import pickle

# Add build directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl
from dataloader import DataLoader
from model import SimpleCNN

def get_predictions(logits_list, num_classes):
    """Simple pure-python argmax."""
    preds = []
    num_samples = len(logits_list) // num_classes
    for i in range(num_samples):
        row = logits_list[i*num_classes : (i+1)*num_classes]
        pred = max(range(len(row)), key=lambda k: row[k])
        preds.append(int(pred))
    return preds

def evaluate(args):
    dataset_path = args.dataset
    weights_path = args.weights
    batch_size = args.batch_size
    test_split = args.test_split
    seed = args.seed
    
    print(f"--- CNN Evaluation on {dataset_path} ---")
    
    # Load dataset
    loader = DataLoader(dataset_path, batch_size=batch_size, shuffle=False, 
                      test_split=test_split, seed=seed)
    loader.set_mode('test')
    
    if len(loader) == 0:
        print("Warning: Test set is empty. Evaluate on full dataset instead.")
        loader = DataLoader(dataset_path, batch_size=batch_size, shuffle=False)
    
    device = dl.Device.GPU
    num_classes = len(loader.classes)
    model = SimpleCNN(num_classes=num_classes)
    model.to(device)
    model.print_stats()
    
    print(f"\nLoading weights from {weights_path}...")
    model.load_weights(weights_path)
    
    criterion = dl.CrossEntropyLoss()
    
    print("\nEvaluating...")
    eval_start = time.time()
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    total_load_time = 0.0
    for images, labels, load_time in loader:
        total_load_time += load_time
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.tolist()[0]
        
        logits_list = logits.tolist()
        labels_list = labels.tolist()
        preds = get_predictions(logits_list, num_classes)
        
        for p, l in zip(preds, labels_list):
            if p == int(l):
                correct += 1
            total += 1
        num_batches += 1
    
    eval_time = time.time() - eval_start
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = (100.0 * correct / total) if total > 0 else 0
    
    print("\n========================================")
    print("          Evaluation Results")
    print("========================================")
    print(f"Total Samples:    {total}")
    print(f"Average Loss:     {avg_loss:.4f}")
    print(f"Accuracy:         {accuracy:.2f}%")
    print(f"Dataset Load Time: {total_load_time:.2f} seconds")
    print(f"Evaluation Time:  {eval_time:.2f} seconds")
    print("========================================")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained CNN model')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    evaluate(args)
