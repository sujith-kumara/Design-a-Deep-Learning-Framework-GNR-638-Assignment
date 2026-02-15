import json
import os
import sys
import time
from custom_NN import *

def train(X_train, y_train, BATCH_SIZE, EPOCHS, LR):
    # 1. Define Architecture 
    # Consistent Architecture: 16 Filters -> 16x16 Output after pool
    out_channels = 16 
    
    layers = [
        Conv2D(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1), 
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        # Flatten size: 16 channels * 16 height * 16 width = 4096
        Linear(in_features=out_channels*16*16, out_features=10) 
    ]
    loss_fn = CrossEntropyLoss()
    
    # 2. Complexity Reporting
    current_h, current_w = 32, 32
    total_params = 0
    total_macs = 0
    total_flops = 0

    for l in layers:
        if hasattr(l, 'get_params'):
            # Pass current H/W to get accurate MACs
            p, m, f = l.get_params(current_h, current_w)
            total_params += p
            total_macs += m
            total_flops += f
            
        # Update dimensions for next layer
        if isinstance(l, Conv2D):
            current_h = (current_h - l.k + 2*l.pad) // l.stride + 1
            current_w = (current_w - l.k + 2*l.pad) // l.stride + 1
        elif isinstance(l, MaxPool2D):
            current_h = current_h // l.stride
            current_w = current_w // l.stride

    print("-" * 30)
    print(f"Model Complexity:")
    print(f"Total Parameters: {total_params}")
    print(f"Total MACs per pass: {total_macs}")
    print(f"Total FLOPs per pass: {total_flops}")
    print("-" * 30)

    # 3. Training Loop
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0
        batches = 0
        
        for batch_X, batch_y in get_batch(X_train, y_train, BATCH_SIZE):
            # --- Forward Pass ---
            out = layers[0].forward(batch_X) # Conv
            out = layers[1].forward(out)     # ReLU
            out = layers[2].forward(out)     # Pool
            
            # Flatten [Batch, 16, 16, 16] -> [Batch, 4096]
            out_flat = flatten_tensor(out)
            
            # Linear & Loss
            logits = layers[3].forward(out_flat)
            loss = loss_fn.forward(logits, batch_y)
            epoch_loss += loss
            batches += 1
            
            # --- Calculate Accuracy ---
            for i in range(len(logits)):
                prediction = logits[i].index(max(logits[i]))
                if prediction == batch_y[i]:
                    total_correct += 1
                total_samples += 1
            
            # --- Backward Pass ---
            grad = loss_fn.backward(batch_y)
            grad = layers[3].backward(grad, LR) # FC Layer
            
            # --- Dynamic Reshape (Inverse Flatten) ---
            grad_reshaped = []
            channels = out_channels # 16
            height = 16             # 32 / 2
            width = 16              # 32 / 2
            
            for b_idx in range(len(grad)):
                flat_g = grad[b_idx]
                channels_g = []
                idx = 0
                for c in range(channels):
                    row_g = []
                    for r in range(height):
                        row_g.append(flat_g[idx : idx + width])
                        idx += width
                    channels_g.append(row_g)
                grad_reshaped.append(channels_g)
            
            # Continue Backprop
            grad = layers[2].backward(grad_reshaped, LR) # Pool
            grad = layers[1].backward(grad, LR)          # ReLU
            grad = layers[0].backward(grad, LR)          # Conv
            
            if batches % 10 == 0:
                current_acc = (total_correct / total_samples) * 100
                print(f"Epoch {epoch+1}, Batch {batches}, Loss: {loss:.4f}, Acc: {current_acc:.2f}%")

        avg_loss = epoch_loss / batches
        epoch_acc = (total_correct / total_samples) * 100
        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, Final Acc: {epoch_acc:.2f}% ---")

    # 4. Save Final Weights
    print("Saving model weights...")
    model_weights = {}
    for i, l in enumerate(layers):
        if isinstance(l, Conv2D):
            model_weights[f"layer_{i}_weights"] = l.filters
            model_weights[f"layer_{i}_bias"] = l.bias
        elif isinstance(l, Linear):
            model_weights[f"layer_{i}_weights"] = l.W
            model_weights[f"layer_{i}_bias"] = l.b
            
    with open("model_weights.json", "w") as f:
        json.dump(model_weights, f)
    print("Weights saved to model_weights.json")

def train_test_split(X, y, test_size=0.2, seed=42):
    random.seed(seed)
    
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # --- CORRECTED SPLIT LOGIC ---
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    print(f"Split complete: {len(X_train)} train, {len(X_test)} test.")
    return X_train, X_test, y_train, y_test