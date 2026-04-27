
import os
import cv2
import numpy as np
from PIL import Image

def get_best_match(base_patch, candidates, overlap_range=(10, 64)):
    best_val = -1
    best_patch_idx = -1
    best_overlap = 0
    best_rotation = 0
    
    # Try right neighbor
    # base_patch right edge vs candidate left edge
    for idx, cand_path in candidates:
        cand = cv2.imread(cand_path)
        for rot in [0, 1, 2, 3]: # 0, 90, 180, 270
            rotated_cand = np.rot90(cand, rot)
            for overlap in range(overlap_range[0], overlap_range[1]):
                # Compare base_patch[:, -overlap:] with rotated_cand[:, :overlap]
                diff = np.mean(np.abs(base_patch[:, -overlap:].astype(float) - rotated_cand[:, :overlap].astype(float)))
                if best_val == -1 or diff < best_val:
                    best_val = diff
                    best_patch_idx = idx
                    best_overlap = overlap
                    best_rotation = rot
    return best_patch_idx, best_overlap, best_rotation, best_val

# Let's just inspect some patches first
patch_dir = "/Users/sujith/Desktop/IITB/GNR_638_project/patches"
p0 = cv2.imread(os.path.join(patch_dir, "patch_0.png"))
print(f"Patch 0 shape: {p0.shape}")

# Sample a few patches to see if they overlap with p0
for i in range(1, 10):
    p_other = cv2.imread(os.path.join(patch_dir, f"patch_{i}.png"))
    # ... logic to check overlap ...
