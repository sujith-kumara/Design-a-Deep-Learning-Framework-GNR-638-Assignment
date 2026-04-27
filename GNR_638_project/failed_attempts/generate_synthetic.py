
import cv2
import numpy as np
import os
import random

def generate_synthetic_patches(image_path, output_dir, grid_size=(15, 15), patch_dim=128, overlap=32):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Base image not found")
        return
    
    # Calculate crop logic to match grid_size and overlap
    # Total width used = 128 + (14 * (128 - 32)) = 128 + 14*96 = 128 + 1344 = 1472
    # Total height used = 128 + 14*96 = 1472
    
    os.makedirs(output_dir, exist_ok=True)
    
    stride = patch_dim - overlap
    
    patch_count = 0
    indices = list(range(grid_size[0] * grid_size[1]))
    random_indices = indices.copy()
    random.shuffle(random_indices) # Shuffle all except patch_0 if we want to mimic competition
    
    # Actually, the competition says patch_0 is anchor.
    # So we keep patch_0 mapped to (0,0).
    
    patch_map = {} # grid_pos -> patch_idx
    
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            y = r * stride
            x = c * stride
            patch = img[y:y+patch_dim, x:x+patch_dim]
            
            # Pad if needed
            if patch.shape[0] < patch_dim or patch.shape[1] < patch_dim:
                patch = cv2.copyMakeBorder(patch, 0, patch_dim-patch.shape[0], 0, patch_dim-patch.shape[1], cv2.BORDER_REFLECT)
            
            # Determine which index this patch takes
            if r == 0 and c == 0:
                idx = 0
            else:
                # Pick a random index from the rest
                idx = random_indices.pop(random_indices.index(0) if 0 in random_indices else 0) # This is a bit clumsy
                # Better: 
                # idx = some random unused index
                pass
    
    # Re-logic for simple shuffle
    all_coords = [(r, c) for r in range(grid_size[0]) for c in range(grid_size[1])]
    anchor = all_coords.pop(0) # (0,0)
    random.shuffle(all_coords)
    
    # (0,0) -> patch_0
    y, x = 0, 0
    patch = img[y:y+patch_dim, x:x+patch_dim]
    cv2.imwrite(os.path.join(output_dir, "patch_0.png"), patch)
    
    # Others -> rest
    for i, (r, c) in enumerate(all_coords):
        idx = i + 1
        y = r * stride
        x = c * stride
        patch = img[y:y+patch_dim, x:x+patch_dim]
        
        # Random rotation
        rot = random.choice([0, 1, 2, 3])
        patch = np.rot90(patch, rot)
        
        cv2.imwrite(os.path.join(output_dir, f"patch_{idx}.png"), patch)

    print(f"Generated {grid_size[0]*grid_size[1]} patches in {output_dir}")

if __name__ == "__main__":
    # Example usage:
    # generate_synthetic_patches("mumbai_map_large.png", "synthetic_patches")
    pass
