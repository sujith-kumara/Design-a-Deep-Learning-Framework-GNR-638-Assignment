
import os
import cv2
import numpy as np
from PIL import Image

class MapStitcher:
    def __init__(self, patch_dir, grid_size=(15, 15), patch_dim=128, overlap_range=(20, 45)):
        self.patch_dir = patch_dir
        self.grid_size = grid_size
        self.patch_dim = patch_dim
        self.overlap_range = overlap_range
        self.patches = self._load_patches()
        self.grid = [[None for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        self.used_patches = set()

    def _load_patches(self):
        patches = {}
        for f in os.listdir(self.patch_dir):
            if f.startswith('patch_') and f.endswith('.png'):
                idx = int(f.split('_')[1].split('.')[0])
                img = cv2.imread(os.path.join(self.patch_dir, f))
                if img is not None:
                    patches[idx] = img
        return patches

    def _compute_match_score(self, anchor_strip, patch_strip):
        # Use Normalized Cross-Correlation (NCC)
        # res has values from -1 to 1. Higher is better.
        res = cv2.matchTemplate(patch_strip, anchor_strip, cv2.TM_CCOEFF_NORMED)
        return res[0][0]

    def _find_best_match(self, left_anchor=None, top_anchor=None):
        best_score = -1.0 # NCC ranges from -1 to 1
        best_match = None # (idx, rotation, ov_x, ov_y)
        
        for idx, patch_orig in self.patches.items():
            if idx in self.used_patches:
                continue
            
            for rot in [0, 1, 2, 3]:
                patch = np.rot90(patch_orig, rot)
                
                ov_x_range = range(self.overlap_range[0], self.overlap_range[1]) if left_anchor is not None else [0]
                ov_y_range = range(self.overlap_range[0], self.overlap_range[1]) if top_anchor is not None else [0]
                
                for ov_x in ov_x_range:
                    score_x = 1.0 # Default if no left neighbor
                    if left_anchor is not None:
                        # Anchor right strip vs Patch left strip
                        score_x = self._compute_match_score(left_anchor[:, -ov_x:], patch[:, :ov_x])
                    
                    if score_x < 0.7 and left_anchor is not None: continue # Skip poor horizontal matches

                    for ov_y in ov_y_range:
                        score_y = 1.0 # Default if no top neighbor
                        if top_anchor is not None:
                            # Anchor bottom strip vs Patch top strip
                            score_y = self._compute_match_score(top_anchor[-ov_y:, :], patch[:ov_y, :])
                        
                        # Stacked Score (Average or Product)
                        total_score = (score_x + score_y) / 2
                        if total_score > best_score:
                            best_score = total_score
                            best_match = (idx, rot, ov_x, ov_y, total_score)
                            if total_score > 0.999: return best_match
        return best_match

    def stitch(self):
        p0 = self.patches[0]
        self.grid[0][0] = (0, 0, 0, 0, 1.0) # idx, rot, ov_x, ov_y, score
        self.used_patches.add(0)
        
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if r == 0 and c == 0: continue
                
                left_anchor = None
                if c > 0 and self.grid[r][c-1] is not None:
                    l_idx, l_rot, _, _, _ = self.grid[r][c-1]
                    left_anchor = np.rot90(self.patches[l_idx], l_rot)
                
                top_anchor = None
                if r > 0 and self.grid[r-1][c] is not None:
                    t_idx, t_rot, _, _, _ = self.grid[r-1][c]
                    top_anchor = np.rot90(self.patches[t_idx], t_rot)
                
                match = self._find_best_match(left_anchor, top_anchor)
                if match:
                    idx, rot, ov_x, ov_y, score = match
                    self.grid[r][c] = (idx, rot, ov_x, ov_y, score)
                    self.used_patches.add(idx)
                    print(f"Matched ({r},{c}): Patch {idx}, Rot {rot}, Overlaps ({ov_x},{ov_y}), Score: {score:.4f}")
                else:
                    # Fallback with relaxed thresholds if needed or just mark as None
                    print(f"WARNING: No match for ({r},{c}), attempting relaxed search...")
                    # For now, let's just avoid the crash
                    self.grid[r][c] = None

    def assemble(self):
        # Calculate coordinates for each patch
        coords = [[(0, 0) for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        # Determine X coordinates for each row (they might vary slightly, we'll take the max or handle per-row)
        # To be safe, we'll use per-patch coordinates relative to their neighbors
        for r in range(self.grid_size[0]):
            for c in range(1, self.grid_size[1]):
                if self.grid[r][c] is not None and self.grid[r][c-1] is not None:
                    ov = self.grid[r][c][2]
                    prev_x = coords[r][c-1][0]
                    coords[r][c] = (prev_x + (self.patch_dim - ov), coords[r][c][1])
                elif self.grid[r][c] is not None:
                     # Default spacing if neighbor is missing
                     coords[r][c] = (coords[r][c-1][0] + 96, coords[r][c][1])
        
        # Determine Y coordinates for each column
        for c in range(self.grid_size[1]):
            for r in range(1, self.grid_size[0]):
                if self.grid[r][c] is not None and self.grid[r-1][c] is not None:
                    ov = self.grid[r][c][3] # Per-patch vertical overlap
                    prev_y = coords[r-1][c][1]
                    coords[r][c] = (coords[r][c][0], prev_y + (self.patch_dim - ov))
                elif self.grid[r][c] is not None:
                     coords[r][c] = (coords[r][c][0], coords[r-1][c][1] + 96)
                
        # Find max dimensions
        max_x = 0
        max_y = 0
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                max_x = max(max_x, coords[r][c][0] + self.patch_dim)
                max_y = max(max_y, coords[r][c][1] + self.patch_dim)
        
        print(f"Canvas size: {max_x}x{max_y}")
        canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        
        # Place patches on canvas (with simple overwriting or averaging)
        # We'll do it in reverse order or use a mask for better blending if needed
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                p_info = self.grid[r][c]
                if p_info is not None:
                    px, py = coords[r][c]
                    img = np.rot90(self.patches[p_info[0]], p_info[1])
                    canvas[py:py+self.patch_dim, px:px+self.patch_dim] = img
                
        return canvas

if __name__ == "__main__":
    stitcher = MapStitcher("patches")
    stitcher.stitch()
    reconstructed = stitcher.assemble()
    cv2.imwrite("reconstructed_map.png", reconstructed)
    print("Reconstruction saved to reconstructed_map.png")
