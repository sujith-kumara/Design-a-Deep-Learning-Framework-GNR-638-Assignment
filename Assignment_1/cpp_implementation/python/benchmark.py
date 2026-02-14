import os
import time
import sys
from dataloader import DataLoader
from model import SimpleCNN

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl

def benchmark():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1')
    if not os.path.exists(data_dir):
         # Try finding any data dir
         for d in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')):
             if d.startswith('data'):
                 data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d)
                 break
    
    batch_size = 64
    print(f"Benchmarking on {data_dir} with batch_size {batch_size}...")
    
    loader = DataLoader(data_dir, batch_size=batch_size, shuffle=True)
    model = SimpleCNN(num_classes=len(loader.classes))
    device = dl.Device.CPU
    model.to(device)
    
    params = model.parameters()
    optimizer = dl.SGD(params, lr=0.01)
    criterion = dl.CrossEntropyLoss()
    
    # Warm up
    it = iter(loader)
    try:
        images, labels, _ = next(it)
    except StopIteration:
        print("No data found!")
        return
    
    print("Starting benchmark for 5 batches...")
    start_time = time.time()
    for i in range(5):
        try:
            images, labels, _ = next(it)
        except StopIteration:
            break
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Batch {i+1} done")
        
    end_time = time.time()
    avg_time = (end_time - start_time) / 5
    print(f"\nAverage time per batch (size {batch_size}): {avg_time:.4f} seconds")
    
    # Calculation
    total_samples = 60000 # Max of data_1/data_2
    num_batches = total_samples // batch_size
    total_epoch_time = num_batches * avg_time
    print(f"Estimated time for 1 epoch (60k images): {total_epoch_time / 60:.2f} minutes ({total_epoch_time / 3600:.2f} hours)")

if __name__ == "__main__":
    benchmark()
