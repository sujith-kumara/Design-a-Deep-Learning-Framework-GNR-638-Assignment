#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import pickle

# Add build directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl
from model import SimpleCNN

def save_feature_maps(activations_list, shape, save_dir, layer_name):
    """Save feature maps as images. activations_list is flat."""
    os.makedirs(save_dir, exist_ok=True)
    
    # shape: [batch, channels, height, width]
    _, c, h, w = shape
    
    # Extract first example in batch
    for i in range(min(c, 32)):  # Save first 32 channels
        channel_data = activations_list[i*(h*w) : (i+1)*(h*w)]
        
        # Normalize to [0, 255] manually
        if not channel_data: continue
        c_min = min(channel_data)
        c_max = max(channel_data)
        denom = (c_max - c_min) + 1e-8
        
        # Reconstruct 2D image for OpenCV
        # This is a bit slow in pure Python but OK for visualization
        img_data = []
        for row in range(h):
            img_row = []
            for col in range(w):
                val = channel_data[row*w + col]
                norm_val = int(255 * (val - c_min) / denom)
                img_row.append(min(255, max(0, norm_val)))
            img_data.append(img_row)
        
        # Convert to a format cv2 understands (list of lists works if passed as list of lists of uint8)
        # However, for simplicity and speed, we can build a raw byte string or similar
        # but the easiest way is just to use standard library or cv2's ability to take nested lists
        # in recent versions, but it's safer to use dummy array if allowed or just avoid.
        # Wait, cv2.imwrite needs an array-like. Since NumPy is banned, we have to be careful.
        # If cv2 is allowed for "loading and basic image processing", maybe it's fine?
        # Let's try to use a dummy image and fill it.
        
        # Actually, let's use standard Python's array or similar? No, just use cv2's from_buffer if possible
        # or just avoid saving if it's too hard without NumPy. 
        # But if user wants visualization, we should try.
        
        # Let's use a simple approach: build a PGM or PPM file manually?
        # That's 100% standard library.
        save_path_pgm = os.path.join(save_dir, f"{layer_name}_channel_{i:02d}.pgm")
        with open(save_path_pgm, 'w') as f:
            f.write(f"P2\n{w} {h}\n255\n")
            for row in img_data:
                f.write(" ".join(map(str, row)) + "\n")
                
    print(f"✓ Saved {min(c, 32)} feature maps to {save_dir} as PGM files")

def visualize_model(args):
    weights_path = args.weights
    image_path = args.image
    output_dir = args.output
    
    # Load image with OpenCV (ALLOWED)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    
    # Process into flat list (Batch=1, C=3, H=32, W=32)
    flat_img = []
    h, w, c = img.shape
    for channel in range(c):
        for row in range(h):
            for col in range(w):
                flat_img.append(float(img[row, col, channel]) / 255.0)
    
    x = dl.Tensor.from_list(flat_img, [1, 3, 32, 32])
    
    # Initialize Model (3-Conv version)
    model = SimpleCNN(num_classes=10)
    if weights_path:
        model.load_weights(weights_path)
    
    print("\nGenerating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Forward pass and save
    x = model.conv1(x)
    x = x.relu()
    save_feature_maps(x.tolist(), x.shape, os.path.join(output_dir, "conv1"), "conv1")
    
    x = x.maxpool2d(2, 2)
    x = model.conv2(x)
    x = x.relu()
    save_feature_maps(x.tolist(), x.shape, os.path.join(output_dir, "conv2"), "conv2")
    
    print(f"\n✓ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CNN features')
    parser.add_argument('--weights', type=str, default='cnn_weights.pkl')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='visualizations')
    args = parser.parse_args()
    visualize_model(args)
