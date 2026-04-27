
import os
import cv2
import numpy as np

class MapStitcherSSD:
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

    def _find_best_match(self, left_anchor=None, top_anchor=None):
        best_diff = float('inf')
        best_match = None
        
        for idx, patch_orig in self.patches.items():
            if idx in self.used_patches:
                continue
            
            for rot in [0, 1, 2, 3]:
                patch = np.rot90(patch_orig, rot)
                
                ov_x_range = range(self.overlap_range[0], self.overlap_range[1]) if left_anchor is not None else [0]
                ov_y_range = range(self.overlap_range[0], self.overlap_range[1]) if top_anchor is not None else [0]
                
                for ov_x in ov_x_range:
                    diff_x = 0
                    if left_anchor is not None:
                        diff_x = np.mean(np.abs(left_anchor[:, -ov_x:] - patch[:, :ov_x]))
                    
                    if diff_x > 30 and left_anchor is not None: continue # Relaxed threshold

                    for ov_y in ov_y_range:
                        diff_y = 0
                        if top_anchor is not None:
                            diff_y = np.mean(np.abs(top_anchor[-ov_y:, :] - patch[:ov_y, :]))
                        
                        total_diff = diff_x + diff_y
                        if total_diff < best_diff:
                            best_diff = total_diff
                            best_match = (idx, rot, ov_x, ov_y, 1.0 - (total_diff/255.0))
                            if total_diff < 0.5: return best_match
        return best_match

    def stitch(self):
        self.grid[0][0] = (0, 0, 0, 0, 1.0)
        self.used_patches.add(0)
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if r == 0 and c == 0: continue
                l = np.rot90(self.patches[self.grid[r][c-1][0]], self.grid[r][c-1][1]) if (c > 0 and self.grid[r][c-1]) else None
                t = np.rot90(self.patches[self.grid[r-1][c][0]], self.grid[r-1][c][1]) if (r > 0 and self.grid[r-1][c]) else None
                match = self._find_best_match(l, t)
                if match:
                    self.grid[r][c] = match
                    self.used_patches.add(match[0])
                    print(f"SSD Matched ({r},{c}): Patch {match[0]}, Score: {match[4]:.4f}")
                else: 
                    print(f"WARNING: No match for ({r},{c})")

    def assemble(self):
        coords = [[(0, 0) for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for r in range(self.grid_size[0]):
            for c in range(1, self.grid_size[1]):
                if self.grid[r][c] and self.grid[r][c-1]:
                    coords[r][c] = (coords[r][c-1][0] + (128 - self.grid[r][c][2]), coords[r][c][1])
                elif self.grid[r][c]: coords[r][c] = (coords[r][c-1][0] + 96, coords[r][c][1])
        for c in range(self.grid_size[1]):
            for r in range(1, self.grid_size[0]):
                if self.grid[r][c] and self.grid[r-1][c]:
                    coords[r][c] = (coords[r][c][0], coords[r-1][c][1] + (128 - self.grid[r][c][3]))
                elif self.grid[r][c]: coords[r][c] = (coords[r][c][0], coords[r-1][c][1] + 96)
        
        mx, my = 0, 0
        for r in range(15):
            for c in range(15):
                if self.grid[r][c]:
                    mx = max(mx, coords[r][c][0]+128); my = max(my, coords[r][c][1]+128)
        
        canvas = np.zeros((my, mx, 3), dtype=np.uint8)
        for r in range(15):
            for c in range(15):
                if self.grid[r][c]:
                    px, py = coords[r][c]
                    canvas[py:py+128, px:px+128] = np.rot90(self.patches[self.grid[r][c][0]], self.grid[r][c][1])
        return canvas

if __name__ == "__main__":
    ms = MapStitcherSSD("patches")
    ms.stitch()
    cv2.imwrite("reconstructed_ssd.png", ms.assemble())
