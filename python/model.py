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
        # Conv2D: in=3, out=16, kernel=3, stride=1, padding=1
        self.conv = dl.Conv2D(3, 16, 3, 1, 1)
        # MaxPool2D(2, 2) reduces 32x32 to 16x16
        # Flattened size: 16 channels * 16 * 16 = 4096
        self.fc = dl.Linear(4096, num_classes)
        
    def forward(self, x):
        # Input x: [B, 3, 32, 32]
        
        # Conv layer
        x = self.conv(x)
        # ReLU (ReLU is a method on Tensor)
        x = x.relu()
        # MaxPool2D (Method on Tensor)
        x = x.maxpool2d(2, 2)
        
        # Flatten: [B, 16, 16, 16] -> [B, 4096]
        # In-place reshape in the current backend
        batch_size = x.shape[0]
        x.reshape([batch_size, 4096])
        
        # Linear layer
        x = self.fc(x)
        
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.conv.parameters() + self.fc.parameters()

    def save_weights(self, path):
        """Save weights to a directory or file using NumPy."""
        weights = {
            'conv_W': self.conv.W.to_numpy(),
            'conv_b': self.conv.b.to_numpy(),
            'fc_W': self.fc.W.to_numpy(),
            'fc_b': self.fc.b.to_numpy()
        }
        np.savez(path, **weights)
        print(f"Weights saved to {path}")

    def load_weights(self, path):
        """Load weights from a .npz file."""
        data = np.load(path)
        self.conv.W = dl.Tensor.from_numpy(data['conv_W'])
        self.conv.b = dl.Tensor.from_numpy(data['conv_b'])
        self.fc.W = dl.Tensor.from_numpy(data['fc_W'])
        self.fc.b = dl.Tensor.from_numpy(data['fc_b'])
        print(f"Weights loaded from {path}")
