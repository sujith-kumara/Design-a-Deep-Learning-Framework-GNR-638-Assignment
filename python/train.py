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

import json

def train(args):
    # Load configuration from file if provided
    config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Merge CLI args with config (CLI args take precedence)
    data_dir = args.dataset
    batch_size = config.get('training', {}).get('batch_size', args.batch_size)
    epochs = config.get('training', {}).get('epochs', args.epochs)
    learning_rate = config.get('training', {}).get('learning_rate', args.lr)
    weights_path = args.save_path
    val_split = args.val_split
    test_split = args.test_split
    seed = args.seed

    print(f"--- CNN Training on {data_dir} ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    print(f"Splits: Val={val_split}, Test={test_split}, Seed={seed}")
    
    # Initialize DataLoader
    try:
        loader = DataLoader(data_dir, batch_size=batch_size, shuffle=True, 
                          val_split=val_split, test_split=test_split, seed=seed)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Device selection
    device = dl.Device.GPU # Try GPU first
    print(f"Device selected: {device}")

    # Initialize Model
    model = SimpleCNN(num_classes=len(loader.classes))
    model.to(device) # Move model to device
    model.print_stats()
    
    # Initialize Optimizer
    params = model.parameters()
    optimizer = dl.SGD(params, lr=learning_rate)
    
    # Loss function
    criterion = dl.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start = time.time()
        
        # --- Training Phase ---
        loader.set_mode('train')
        model_train_start = time.time()
        
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_idx = 0
        
        for images, labels, load_time in loader:
            batch_idx += 1
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_train_loss += loss.to_numpy()[0]
            logits_np = logits.to_numpy()
            labels_np = labels.to_numpy().astype(int)
            predictions = np.argmax(logits_np, axis=1)
            train_correct += np.sum(predictions == labels_np)
            train_total += labels_np.shape[0]

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"  [Train] Batch {batch_idx}/{len(loader)}, Loss: {loss.to_numpy()[0]:.4f}, Acc: {100.0 * train_correct / train_total:.2f}%", flush=True)

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = total_train_loss / len(loader)

        # --- Validation Phase ---
        val_acc = 0.0
        avg_val_loss = 0.0
        if val_split > 0:
            loader.set_mode('val')
            total_val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            print("  Running Validation...")
            for images, labels, load_time in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                total_val_loss += loss.to_numpy()[0]
                logits_np = logits.to_numpy()
                labels_np = labels.to_numpy().astype(int)
                predictions = np.argmax(logits_np, axis=1)
                val_correct += np.sum(predictions == labels_np)
                val_total += labels_np.shape[0]
            
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = total_val_loss / len(loader)
            print(f"  [Val] Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = weights_path.replace('.npz', '_best.npz')
                model.save_weights(best_path)
                print(f"  ✓ Best model saved with {val_acc:.2f}% accuracy")

        epoch_end = time.time()
        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_split > 0:
            print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  Total Time: {epoch_end - epoch_start:.2f} seconds")

    print(f"\nFinal training loss: {avg_train_loss:.4f}, acc: {train_acc:.2f}%")
    
    # --- Final Evaluation on Test Set ---
    print("\n" + "="*40)
    print("      FINAL EVALUATION (TEST SET)")
    print("="*40)
    loader.set_mode('test')
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        test_loss += loss.to_numpy()[0]
        predictions = np.argmax(logits.to_numpy(), axis=1)
        test_correct += np.sum(predictions == labels.to_numpy())
        test_total += labels.shape[0]
    
    print(f"Test Loss: {test_loss/len(loader):.4f}")
    print(f"Test Acc:  {100.0 * test_correct / test_total:.2f}%")
    print("="*40)

    # Final save
    print(f"\nSaving final weights to {weights_path}")
    model.save_weights(weights_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN on a custom dataset')
    parser.add_argument('--dataset', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1'),
                        help='Path to the dataset directory')
    parser.add_argument('--config', type=str, help='Path to model configuration file (JSON)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.0, help='Fraction of data for validation')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data for final test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_weights.npz'),
                        help='Path to save weights')
    
    args = parser.parse_args()
    train(args)

