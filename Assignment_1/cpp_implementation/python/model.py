import os
import sys
import pickle

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl

class SimpleCNN:
    def __init__(self, num_classes=10):
        # Block 1: 3 → 32 channels, 3x3, P=1, S=1 (Output: 32x32)
        self.conv1 = dl.Conv2D(3, 32, 3, 1, 1)
        # After MaxPool: 16x16
        
        # Block 2: 32 → 64 channels, 3x3, P=1, S=1 (Output: 16x16)
        self.conv2 = dl.Conv2D(32, 64, 3, 1, 1)
        # After MaxPool: 8x8
        
        # Block 3: 64 → 128 channels, 3x3, P=1, S=1 (Output: 8x8)
        self.conv3 = dl.Conv2D(64, 128, 3, 1, 1)
        # After MaxPool: 4x4
        # Flattened size: 128 * 4 * 4 = 2048
        
        # Classifier
        self.fc1 = dl.Linear(2048, 128)
        self.fc2 = dl.Linear(128, num_classes)
        self.num_classes = num_classes
        self.device = dl.Device.CPU

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.conv1.W = self.conv1.W.to(device)
        self.conv1.b = self.conv1.b.to(device)
        self.conv2.W = self.conv2.W.to(device)
        self.conv2.b = self.conv2.b.to(device)
        self.conv3.W = self.conv3.W.to(device)
        self.conv3.b = self.conv3.b.to(device)
        self.fc1.W = self.fc1.W.to(device)
        self.fc1.b = self.fc1.b.to(device)
        self.fc2.W = self.fc2.W.to(device)
        self.fc2.b = self.fc2.b.to(device)
        return self

    def forward(self, x):
        # Block 1
        x = self.conv1(x)           # [B, 32, 32, 32]
        x = x.relu()
        x = x.maxpool2d(2, 2)       # [B, 32, 16, 16]
        
        # Block 2
        x = self.conv2(x)           # [B, 64, 16, 16]
        x = x.relu()
        x = x.maxpool2d(2, 2)       # [B, 64, 8, 8]
        
        # Block 3
        x = self.conv3(x)           # [B, 128, 8, 8]
        x = x.relu()
        x = x.maxpool2d(2, 2)       # [B, 128, 4, 4]
        
        # Flatten
        batch_size = x.shape[0]
        x.reshape([batch_size, 2048])
        
        # Classifier
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return (self.conv1.parameters() + self.conv2.parameters() + 
                self.conv3.parameters() + self.fc1.parameters() + 
                self.fc2.parameters())

    def save_weights(self, path):
        """Save weights to a file using pickle."""
        weights = {
            'conv1_W_data': self.conv1.W.tolist(), 'conv1_W_shape': self.conv1.W.shape,
            'conv1_b_data': self.conv1.b.tolist(), 'conv1_b_shape': self.conv1.b.shape,
            'conv2_W_data': self.conv2.W.tolist(), 'conv2_W_shape': self.conv2.W.shape,
            'conv2_b_data': self.conv2.b.tolist(), 'conv2_b_shape': self.conv2.b.shape,
            'conv3_W_data': self.conv3.W.tolist(), 'conv3_W_shape': self.conv3.W.shape,
            'conv3_b_data': self.conv3.b.tolist(), 'conv3_b_shape': self.conv3.b.shape,
            'fc1_W_data': self.fc1.W.tolist(), 'fc1_W_shape': self.fc1.W.shape,
            'fc1_b_data': self.fc1.b.tolist(), 'fc1_b_shape': self.fc1.b.shape,
            'fc2_W_data': self.fc2.W.tolist(), 'fc2_W_shape': self.fc2.W.shape,
            'fc2_b_data': self.fc2.b.tolist(), 'fc2_b_shape': self.fc2.b.shape
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Weights saved to {path} using pickle.")

    def load_weights(self, path):
        """Load weights using pickle."""
        if not os.path.exists(path):
            print(f"Warning: Weights file {path} not found.")
            return
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.conv1.W = dl.Tensor.from_list(data['conv1_W_data'], data['conv1_W_shape'])
        self.conv1.b = dl.Tensor.from_list(data['conv1_b_data'], data['conv1_b_shape'])
        self.conv2.W = dl.Tensor.from_list(data['conv2_W_data'], data['conv2_W_shape'])
        self.conv2.b = dl.Tensor.from_list(data['conv2_b_data'], data['conv2_b_shape'])
        self.conv3.W = dl.Tensor.from_list(data['conv3_W_data'], data['conv3_W_shape'])
        self.conv3.b = dl.Tensor.from_list(data['conv3_b_data'], data['conv3_b_shape'])
        self.fc1.W = dl.Tensor.from_list(data['fc1_W_data'], data['fc1_W_shape'])
        self.fc1.b = dl.Tensor.from_list(data['fc1_b_data'], data['fc1_b_shape'])
        self.fc2.W = dl.Tensor.from_list(data['fc2_W_data'], data['fc2_W_shape'])
        self.fc2.b = dl.Tensor.from_list(data['fc2_b_data'], data['fc2_b_shape'])
        print(f"Weights loaded from {path} using pickle.")

    def print_stats(self):
        print("\n" + "="*40)
        print("          Model Statistics (3-Conv)          ")
        print("="*40)
        
        # Conv1: (3->32, 3x3)
        c1_p = (3 * 3 * 3 + 1) * 32
        c1_m = (32 * 32 * 32) * (3 * 3 * 3)
        
        # Conv2: (32->64, 3x3)
        c2_p = (3 * 3 * 32 + 1) * 64
        c2_m = (16 * 16 * 64) * (3 * 3 * 32)
        
        # Conv3: (64->128, 3x3)
        c3_p = (3 * 3 * 64 + 1) * 128
        c3_m = (8 * 8 * 128) * (3 * 3 * 64)
        
        # FC1: (2048->128)
        f1_p = (2048 + 1) * 128
        f1_m = 2048 * 128
        
        # FC2: (128->num_classes)
        f2_p = (128 + 1) * self.num_classes
        f2_m = 128 * self.num_classes
        
        total_p = c1_p + c2_p + c3_p + f1_p + f2_p
        total_m = c1_m + c2_m + c3_m + f1_m + f2_m
        
        print(f"Conv1 (32):       Params: {c1_p:,}")
        print(f"Conv2 (64):       Params: {c2_p:,}")
        print(f"Conv3 (128):      Params: {c3_p:,}")
        print(f"FC1 (128):        Params: {f1_p:,}")
        print(f"FC2 ({self.num_classes}):         Params: {f2_p:,}")
        print("-" * 40)
        print(f"TOTAL PARAMS:  {total_p:,}")
        print(f"TOTAL MACs:    {total_m:,}")
        print(f"TOTAL FLOPs:   {2 * total_m:,}")
        print("="*40 + "\n")
