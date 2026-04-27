
import os
import cv2
import numpy as np

def find_neighbors(anchor_idx, patch_dir, all_patches, used_patches, overlap_range=(20, 60)):
    anchor = cv2.imread(os.path.join(patch_dir, f"patch_{anchor_idx}.png"))
    h, w, _ = anchor.shape
    
    results = []
    
    for i in all_patches:
        if i in used_patches:
            continue
        
        cand_orig = cv2.imread(os.path.join(patch_dir, f"patch_{i}.png"))
        if cand_orig is None: continue
        
        for rot in [0, 1, 2, 3]:
            cand = np.rot90(cand_orig, rot)
            
            # Check Right
            for ov in range(overlap_range[0], overlap_range[1]):
                diff = np.mean(np.abs(anchor[:, -ov:].astype(float) - cand[:, :ov].astype(float)))
                if diff < 5: # Threshold for match
                    results.append(('right', i, rot, ov, diff))
                    break
            
            # Check Bottom
            for ov in range(overlap_range[0], overlap_range[1]):
                diff = np.mean(np.abs(anchor[-ov:, :].astype(float) - cand[:ov, :].astype(float)))
                if diff < 5: # Threshold for match
                    results.append(('bottom', i, rot, ov, diff))
                    break
    return results

patch_dir = "patches"
all_patches = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(patch_dir) if f.startswith('patch_')]
used_patches = {0}

neighbors = find_neighbors(0, patch_dir, all_patches, used_patches)
print(f"Neighbors of patch_0: {neighbors}")
