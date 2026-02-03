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

    def print_stats(self):
        """Bonus A: Compute and print FLOPs, MACs, and Parameter counts."""
        print("\n" + "="*40)
        print("          Model Statistics          ")
        print("="*40)
        
        total_params = 0
        total_macs = 0
        total_flops = 0
        
        # --- Layer 1: Conv2D (3, 16, 3, 1, 1) ---
        # Input: [3, 32, 32] -> Output: [16, 32, 32]
        k = 3
        cin = 3
        cout = 16
        h_out, w_out = 32, 32
        
        # Params: (k*k*cin + 1) * cout
        conv_params = (k * k * cin + 1) * cout
        
        # MACs: h_out * w_out * k * k * cin * cout
        conv_macs = h_out * w_out * k * k * cin * cout
        
        # FLOPs: approx 2 * MACs (multiply + add)
        conv_flops = 2 * conv_macs
        
        total_params += conv_params
        total_macs += conv_macs
        total_flops += conv_flops
        
        print(f"Conv2D:")
        print(f"  Params: {conv_params:,}")
        print(f"  MACs:   {conv_macs:,}")
        print(f"  FLOPs:  {conv_flops:,}")

        # --- Activation: ReLU ---
        # Input: [16, 32, 32]
        # FLOPs: 1 comparison per element
        relu_flops = 16 * 32 * 32
        total_flops += relu_flops
        
        # --- Pooling: MaxPool2D (2, 2) ---
        # Input: [16, 32, 32] -> Output: [16, 16, 16]
        # FLOPs: 1 comparison per element of input (roughly)
        pool_flops = 16 * 32 * 32
        total_flops += pool_flops

        # --- Layer 2: Linear (4096 -> 10) ---
        fin = 4096
        fout = 10
        
        # Params: (fin + 1) * fout
        fc_params = (fin + 1) * fout
        
        # MACs: fin * fout
        fc_macs = fin * fout
        
        # FLOPs: 2 * MACs
        fc_flops = 2 * fc_macs
        
        total_params += fc_params
        total_macs += fc_macs
        total_flops += fc_flops

        print(f"Linear:")
        print(f"  Params: {fc_params:,}")
        print(f"  MACs:   {fc_macs:,}")
        print(f"  FLOPs:  {fc_flops:,}")
        
        print("-" * 40)
        print(f"TOTAL PARAMS: {total_params:,}")
        print(f"TOTAL MACs:   {total_macs:,}")
        print(f"TOTAL FLOPs:  {total_flops:,}")
        print("="*40 + "\n")
