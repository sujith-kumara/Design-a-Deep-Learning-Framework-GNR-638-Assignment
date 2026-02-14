import os
import sys
import numpy as np

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl

class SimpleCNN:
    def __init__(self, num_classes=10):
        # First Conv Block: 3 → 32 channels
        self.conv1 = dl.Conv2D(3, 32, 3, 1, 1)
        # After maxpool: 32x32 → 16x16
        
        # Second Conv Block: 32 → 64 channels
        self.conv2 = dl.Conv2D(32, 64, 3, 1, 1)
        # After maxpool: 16x16 → 8x8
        # Flattened size: 64 * 8 * 8 = 4096
        
        # Fully connected layers
        self.fc1 = dl.Linear(4096, 256)
        self.fc2 = dl.Linear(256, num_classes)
        self.num_classes = num_classes
        self.device = dl.Device.CPU

    def to(self, device):
        """Move model to device."""
        self.device = device
        # Move all parameters
        self.conv1.W = self.conv1.W.to(device)
        self.conv1.b = self.conv1.b.to(device)
        self.conv2.W = self.conv2.W.to(device)
        self.conv2.b = self.conv2.b.to(device)
        self.fc1.W = self.fc1.W.to(device)
        self.fc1.b = self.fc1.b.to(device)
        self.fc2.W = self.fc2.W.to(device)
        self.fc2.b = self.fc2.b.to(device)
        return self

        
    def forward(self, x):
        # Input: [B, 3, 32, 32]
        
        # First Conv Block
        x = self.conv1(x)           # [B, 32, 32, 32]
        x = x.relu()                # ReLU activation
        x = x.maxpool2d(2, 2)       # [B, 32, 16, 16]
        
        # Second Conv Block
        x = self.conv2(x)           # [B, 64, 16, 16]
        x = x.relu()                # ReLU activation
        x = x.maxpool2d(2, 2)       # [B, 64, 8, 8]
        
        # Flatten for fully connected layers
        batch_size = x.shape[0]
        x.reshape([batch_size, 4096])  # [B, 4096]
        
        # First FC layer with ReLU
        x = self.fc1(x)             # [B, 256]
        x = x.relu()
        
        # Output layer
        x = self.fc2(x)             # [B, num_classes]
        
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return (self.conv1.parameters() + self.conv2.parameters() + 
                self.fc1.parameters() + self.fc2.parameters())

    def save_weights(self, path):
        """Save weights to a directory or file using NumPy."""
        weights = {
            'conv1_W': self.conv1.W.to_numpy(),
            'conv1_b': self.conv1.b.to_numpy(),
            'conv2_W': self.conv2.W.to_numpy(),
            'conv2_b': self.conv2.b.to_numpy(),
            'fc1_W': self.fc1.W.to_numpy(),
            'fc1_b': self.fc1.b.to_numpy(),
            'fc2_W': self.fc2.W.to_numpy(),
            'fc2_b': self.fc2.b.to_numpy()
        }
        np.savez(path, **weights)
        print(f"Weights saved to {path}")

    def load_weights(self, path):
        """Load weights from a .npz file."""
        data = np.load(path)
        self.conv1.W = dl.Tensor.from_numpy(data['conv1_W'])
        self.conv1.b = dl.Tensor.from_numpy(data['conv1_b'])
        self.conv2.W = dl.Tensor.from_numpy(data['conv2_W'])
        self.conv2.b = dl.Tensor.from_numpy(data['conv2_b'])
        self.fc1.W = dl.Tensor.from_numpy(data['fc1_W'])
        self.fc1.b = dl.Tensor.from_numpy(data['fc1_b'])
        self.fc2.W = dl.Tensor.from_numpy(data['fc2_W'])
        self.fc2.b = dl.Tensor.from_numpy(data['fc2_b'])
        print(f"Weights loaded from {path}")

    def print_stats(self):
        """Compute and print FLOPs, MACs, and Parameter counts."""
        print("\n" + "="*40)
        print("          Model Statistics (2-Conv)          ")
        print("="*40)
        
        # Conv1: (3->32, 3x3, output 32x32)
        conv1_params = (3 * 3 * 3 + 1) * 32
        conv1_macs = (32 * 32 * 32) * (3 * 3 * 3)
        
        # Conv2: (32->64, 3x3, output 16x16)
        conv2_params = (3 * 3 * 32 + 1) * 64
        conv2_macs = (16 * 16 * 64) * (3 * 3 * 32)
        
        # FC1: (4096->256)
        fc1_params = (4096 + 1) * 256
        fc1_macs = 4096 * 256
        
        # FC2: (256->num_classes)
        num_classes = self.num_classes
        fc2_params = (256 + 1) * num_classes
        fc2_macs = 256 * num_classes
        
        total_params = conv1_params + conv2_params + fc1_params + fc2_params
        total_macs = conv1_macs + conv2_macs + fc1_macs + fc2_macs
        total_flops = 2 * total_macs
        
        print(f"Conv1 (3->32):    Params: {conv1_params:,}")
        print(f"Conv2 (32->64):   Params: {conv2_params:,}")
        print(f"FC1 (4096->256):  Params: {fc1_params:,}")
        print(f"FC2 (256->{num_classes}):  Params: {fc2_params:,}")
        print("-" * 40)
        print(f"TOTAL PARAMS:  {total_params:,}")
        print(f"TOTAL MACs:    {total_macs:,}")
        print(f"TOTAL FLOPs:   {total_flops:,}")
        print("="*40 + "\n")
