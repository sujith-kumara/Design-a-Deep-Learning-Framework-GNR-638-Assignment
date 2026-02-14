import os
import time
import sys
import json
import pickle
from dataloader import DataLoader
from model import SimpleCNN

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl

def get_accuracy(logits_list, labels_list, batch_size, num_classes):
    """Simple pure-python accuracy calculation."""
    correct = 0
    for i in range(len(labels_list)):
        # Extract the row for this batch element
        row = logits_list[i*num_classes : (i+1)*num_classes]
        # Find index of max value
        pred = max(range(len(row)), key=lambda k: row[k])
        if int(pred) == int(labels_list[i]):
            correct += 1
    return correct

def train(args):
    # Load configuration from file if provided
    config = {}
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Merge CLI args with config
    data_dir = args.dataset
    batch_size = config.get('training', {}).get('batch_size', args.batch_size)
    epochs = config.get('training', {}).get('epochs', args.epochs)
    learning_rate = config.get('training', {}).get('learning_rate', args.lr)
    # Default to .pkl extension
    weights_path = args.save_path.replace('.npz', '.pkl')
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
    device = dl.Device.GPU
    print(f"Device selected: {device}")

    # Initialize Model
    num_classes = len(loader.classes)
    model = SimpleCNN(num_classes=num_classes)
    model.to(device)
    model.print_stats()
    
    # Initialize Optimizer
    optimizer = dl.SGD(model.parameters(), lr=learning_rate)
    criterion = dl.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start = time.time()
        
        # --- Training Phase ---
        loader.set_mode('train')
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_idx = 0
        total_load_time = 0.0
        
        for images, labels, load_time in loader:
            batch_idx += 1
            total_load_time += load_time
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics using list conversion (tolist instead of to_numpy)
            current_loss = loss.tolist()[0]
            total_train_loss += current_loss
            
            logits_list = logits.tolist()
            labels_list = labels.tolist()
            
            num_in_batch = len(labels_list)
            correct = get_accuracy(logits_list, labels_list, num_in_batch, num_classes)
            train_correct += correct
            train_total += num_in_batch

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                acc = 100.0 * train_correct / train_total
                avg_load = total_load_time / batch_idx
                print(f"  [Train] Batch {batch_idx}/{len(loader)}, Loss: {current_loss:.4f}, Acc: {acc:.2f}%, Load: {avg_load:.4f}s", flush=True)

        avg_train_loss = total_train_loss / len(loader)
        train_acc = 100.0 * train_correct / train_total

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
                
                total_val_loss += loss.tolist()[0]
                vl_logits = logits.tolist()
                vl_labels = labels.tolist()
                
                val_correct += get_accuracy(vl_logits, vl_labels, len(vl_labels), num_classes)
                val_total += len(vl_labels)
            
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = total_val_loss / len(loader)
            print(f"  [Val] Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = weights_path.replace('.pkl', '_best.pkl')
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
        
        tl_list = loss.tolist()
        test_loss += tl_list[0]
        
        t_logits = logits.tolist()
        t_labels = labels.tolist()
        test_correct += get_accuracy(t_logits, t_labels, len(t_labels), num_classes)
        test_total += len(t_labels)
    
    print(f"Test Loss: {test_loss/len(loader):.4f}")
    print(f"Test Acc:  {100.0 * test_correct / test_total:.2f}%")
    print("="*40)

    # Final save
    print(f"\nSaving final weights to {weights_path}")
    model.save_weights(weights_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN on a custom dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--val_split', type=float, default=0.0)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='cnn_weights.pkl')
    
    args = parser.parse_args()
    train(args)
