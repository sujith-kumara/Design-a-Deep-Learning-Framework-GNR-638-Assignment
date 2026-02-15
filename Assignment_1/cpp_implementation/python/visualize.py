#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import pickle
import zlib
import struct

# Add build directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl
from model import SimpleCNN

def write_png(buf, width, height, is_color=False):
    """Pure Python PNG writer using zlib/struct (No NumPy)."""
    # buf is a list of [R,G,B] or grayscale [I]
    mode = 3 if is_color else 0 # 0=Grayscale, 3=Indexed/Palette not used, 2=RGB
    # We'll use RGB (color_type 2) or Grayscale (color_type 0)
    color_type = 2 if is_color else 0
    bit_depth = 8
    
    # PNG signature
    png_sig = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    def make_chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    
    ihdr = make_chunk(b"IHDR", ihdr_data)
    
    # IDAT chunk (pixel data)
    # Each row must start with a filter type byte (0)
    stride = width * (3 if is_color else 1)
    rows = []
    for i in range(height):
        row_data = bytes(buf[i*stride : (i+1)*stride])
        rows.append(b'\x00' + row_data)
    
    idat = make_chunk(b"IDAT", zlib.compress(b"".join(rows)))
    iend = make_chunk(b"IEND", b"")
    
    return png_sig + ihdr + idat + iend

def save_feature_maps(activations_list, shape, save_dir, layer_name):
    """Save feature maps as PNG images. activations_list is flat."""
    os.makedirs(save_dir, exist_ok=True)
    
    # shape: [batch, channels, height, width]
    _, c, h, w = shape
    
    # Extract first example in batch
    for i in range(min(c, 32)):  # Save first 32 channels
        channel_data = activations_list[i*(h*w) : (i+1)*(h*w)]
        
        if not channel_data: continue
        c_min = min(channel_data)
        c_max = max(channel_data)
        denom = (c_max - c_min) + 1e-8
        
        # Build 1D list of normalized pixels [0, 255]
        pixels = [int(255 * (val - c_min) / denom) for val in channel_data]
        
        # 1. Save Grayscale PNG
        save_path_png = os.path.join(save_dir, f"{layer_name}_channel_{i:02d}.png")
        with open(save_path_png, 'wb') as f:
            f.write(write_png(pixels, w, h, is_color=False))

        # 2. Save Heatmap PNG (JET-like)
        heatmap_pixels = []
        for val in pixels:
            # R: high for high val, G: high for mid val, B: high for low val
            r = min(255, max(0, int(255 * (val - 128) / 128 * 2))) if val > 128 else 0
            g = min(255, int(255 * (1 - abs(val - 128) / 128)))
            b = min(255, max(0, int(255 * (128 - val) / 128 * 2))) if val < 128 else 0
            heatmap_pixels.extend([r, g, b])
            
        save_path_heatmap = os.path.join(save_dir, f"{layer_name}_channel_{i:02d}_heatmap.png")
        with open(save_path_heatmap, 'wb') as f:
            f.write(write_png(heatmap_pixels, w, h, is_color=True))
                
    print(f"✓ Saved {min(c, 32)} PNG feature maps and heatmaps to {save_dir}")

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
    
    # Initialize Model (2nd Conv version)
    model = SimpleCNN(num_classes=args.num_classes)
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
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    visualize_model(args)
