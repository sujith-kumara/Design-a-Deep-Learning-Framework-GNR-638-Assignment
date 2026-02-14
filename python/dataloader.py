import os
import cv2
import numpy as np
import time
import sys

# Try to find the deeplearn module
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

try:
    import deeplearn as dl
except ImportError:
    print(f"Warning: Could not import deeplearn from {build_dir}. Ensure the bindings are built.")
    dl = None

class DataLoader:
    def __init__(self, root_dir, batch_size=32, target_size=(32, 32), shuffle=True, 
                 val_split=0.0, test_split=0.0, seed=42):
        init_start_time = time.time()
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.mode = 'train'  # Default mode
        
        # Discover classes and images
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for f in os.listdir(cls_dir):
                if f.lower().endswith('.png') and not f.startswith('.'):
                    self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls_name]))
        
        if not self.samples:
            raise ValueError(f"No PNG images found in {root_dir}")
        
        # Split into train, validation, and test
        total_samples = len(self.samples)
        np.random.seed(self.seed)
        indices = np.random.permutation(total_samples)
        
        test_size = int(total_samples * self.test_split)
        val_size = int(total_samples * self.val_split)
        
        self.test_indices = indices[:test_size]
        self.val_indices = indices[test_size:test_size + val_size]
        self.train_indices = indices[test_size + val_size:]
        
        self.loading_time = time.time() - init_start_time
        print(f"Found {total_samples} images across {len(self.classes)} classes.")
        if self.val_split > 0 or self.test_split > 0:
            print(f"Split: Train={len(self.train_indices)}, Val={len(self.val_indices)}, Test={len(self.test_indices)} (seed={self.seed})")
        print(f"Dataset initialized in {self.loading_time:.4f} seconds.")
    
    def set_mode(self, mode='train'):
        """Set whether to use training, validation, or test data."""
        valid_modes = ['train', 'val', 'validation', 'test']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        
        if mode == 'validation': mode = 'val'
        self.mode = mode

    def __len__(self):
        if self.mode == 'test':
            active_indices = self.test_indices
        elif self.mode == 'val':
            active_indices = self.val_indices
        else:
            active_indices = self.train_indices
        return (len(active_indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        start_time = time.time()
        
        # Determine which indices to use
        if self.mode == 'test':
            active_indices = self.test_indices.copy()
        elif self.mode == 'val':
            active_indices = self.val_indices.copy()
        else:
            active_indices = self.train_indices.copy()
        
        # Set seed for reproducibility
        np.random.seed(self.seed)
        
        if self.shuffle and self.mode == 'train':
            # Only shuffle training data
            np.random.shuffle(active_indices)
            
        for i in range(0, len(active_indices), self.batch_size):
            batch_indices = active_indices[i:i + self.batch_size]
            batch_samples = [self.samples[idx] for idx in batch_indices]
            
            batch_start = time.time()
            images = []
            labels = []
            
            for img_path, label in batch_samples:
                # Load image with OpenCV
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, self.target_size)
                
                # Convert BGR to RGB if needed, here we'll keep as is or normalize
                img = img.astype(np.float32) / 255.0
                
                # Reshape to (C, H, W) for common DL format if needed
                # OpenCV is (H, W, C)
                img = np.transpose(img, (2, 0, 1))
                
                images.append(img)
                labels.append(label)
            
            if not images:
                continue
                
            # Convert to NumPy arrays
            batch_images_np = np.stack(images).astype(np.float32)
            batch_labels_np = np.array(labels, dtype=np.float32)
            
            # Convert to backend Tensor
            if dl:
                batch_images_tensor = dl.Tensor.from_numpy(batch_images_np)
                batch_labels_tensor = dl.Tensor.from_numpy(batch_labels_np)
            else:
                batch_images_tensor = batch_images_np
                batch_labels_tensor = batch_labels_np
            
            batch_end = time.time()
            loading_time = batch_end - batch_start
            
            yield batch_images_tensor, batch_labels_tensor, loading_time

        total_time = time.time() - start_time
        print(f"Finished Epoch. Total loading time: {total_time:.4f} seconds.")

if __name__ == "__main__":
    # Quick sanity check if run directly
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        loader = DataLoader(data_dir, batch_size=64)
        for images, labels, load_time in loader:
            print(f"Batch loaded in {load_time:.6f}s. Shape: {images.shape if hasattr(images, 'shape') else 'N/A'}")
            break
