#!/usr/bin/env python3
"""
Visualization script for CNN features and activations.
Usage: python3 visualize.py --weights <path> --image <path>
"""
import argparse
import os
import sys
import numpy as np
import cv2

# Add build directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, '..', 'build', 'bindings')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import deeplearn as dl
from model import SimpleCNN


def save_feature_maps(activations, save_dir, layer_name):
    """Save feature maps as images."""
    os.makedirs(save_dir, exist_ok=True)
    
    # activations shape: [channels, height, width]
    num_channels = activations.shape[0]
    
    # Normalize and save each channel
    for i in range(min(num_channels, 32)):  # Save first 32 channels
        feat_map = activations[i]
        
        # Normalize to [0, 255]
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
        feat_map = (feat_map * 255).astype(np.uint8)
        
        # Apply colormap for better visualization
        feat_map_color = cv2.applyColorMap(feat_map, cv2.COLORMAP_JET)
        
        save_path = os.path.join(save_dir, f"{layer_name}_channel_{i:02d}.png")
        cv2.imwrite(save_path, feat_map_color)
    
    print(f"✓ Saved {min(num_channels, 32)} feature maps to {save_dir}")


def visualize_model(args):
    weights_path = args.weights
    image_path = args.image
    output_dir = args.output
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32, 32))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    # Convert to tensor
    x = dl.Tensor.from_numpy(img_batch)
    
    # Load model
    model = SimpleCNN(num_classes=10)  # Adjust as needed
    if weights_path:
        model.load_weights(weights_path)
        print("Weights loaded")
    
    print("\nGenerating visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Forward pass with intermediate activations
    # Conv1
    x = model.conv1(x)
    x = x.relu()
    conv1_act = x.to_numpy()[0]  # Remove batch dim
    save_feature_maps(conv1_act, os.path.join(output_dir, "conv1"), "conv1")
    
    x = x.maxpool2d(2, 2)
    
    # Conv2
    x = model.conv2(x)
    x = x.relu()
    conv2_act = x.to_numpy()[0]
    save_feature_maps(conv2_act, os.path.join(output_dir, "conv2"), "conv2")
    
    # Save input image with same size as feature maps for comparison
    input_resized = cv2.resize(img_rgb, (32, 32))
    cv2.imwrite(os.path.join(output_dir, "input_image.png"), cv2.cvtColor(input_resized, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"  - Input image: input_image.png")
    print(f"  - Conv1 features: conv1/")
    print(f"  - Conv2 features: conv2/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CNN features')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to weights file (.npz, optional)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='visualizations',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    visualize_model(args)
