import os
import sys
import time
from dataloader import DataLoader

def test_on_dataset(dataset_path, name):
    print(f"\n--- Testing DataLoader on {name} ({dataset_path}) ---")
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Skipping.")
        return

    try:
        loader = DataLoader(dataset_path, batch_size=32, shuffle=True)
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        return

    batch_times = []
    num_batches = 0
    total_samples = 0
    
    # Iterate through one full epoch to measure total time
    start_total = time.time()
    for images, labels, load_time in loader:
        batch_times.append(load_time)
        num_batches += 1
        # If it's a dl.Tensor, it has a shape property (which is a list from pybind11)
        # If it's a numpy array, it has a shape tuple
        total_samples += images.shape[0] if hasattr(images, 'shape') else 0
        
        if num_batches == 1:
            print(f"First batch shape: {images.shape if hasattr(images, 'shape') else 'N/A'}")
            print(f"First labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")

    end_total = time.time()
    
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"Processed {num_batches} batches ({total_samples} samples).")
        print(f"Average batch loading time: {avg_batch_time:.6f} seconds.")
        print(f"Total epoch loading time (including iteration overhead): {end_total - start_total:.4f} seconds.")
    else:
        print("No batches were processed.")

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    
    test_on_dataset(os.path.join(base_path, 'data_1'), "data_1")
    test_on_dataset(os.path.join(base_path, 'data_2'), "data_2")
