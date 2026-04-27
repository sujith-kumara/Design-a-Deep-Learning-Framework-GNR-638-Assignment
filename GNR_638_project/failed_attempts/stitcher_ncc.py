
import os
import cv2
import numpy as np

class MapStitcherNCC:
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
            if f.endswith('.png'):
                idx = int(''.join(filter(str.isdigit, f)))
                img = cv2.imread(os.path.join(self.patch_dir, f))
                if img is not None: patches[idx] = img
        return patches

    def _find_best_match(self, left=None, top=None):
        best_score = -1.0
        best_match = None
        for idx, p_orig in self.patches.items():
            if idx in self.used_patches: continue
            for rot in [0, 1, 2, 3]:
                patch = np.rot90(p_orig, rot)
                ox_rng = range(self.overlap_range[0], self.overlap_range[1]) if left is not None else [0]
                oy_rng = range(self.overlap_range[0], self.overlap_range[1]) if top is not None else [0]
                for ox in ox_rng:
                    s_x = cv2.matchTemplate(patch[:, :ox], left[:, -ox:], cv2.TM_CCOEFF_NORMED)[0][0] if left is not None else 1.0
                    if s_x < 0.4 and left is not None: continue # Relaxed NCC threshold
                    for oy in oy_rng:
                        s_y = cv2.matchTemplate(patch[:oy, :], top[-oy:, :], cv2.TM_CCOEFF_NORMED)[0][0] if top is not None else 1.0
                        score = (s_x + s_y) / 2
                        if score > best_score:
                            best_score = score
                            best_match = (idx, rot, ox, oy, score)
                            if score > 0.999: return best_match
        return best_match

    def stitch(self):
        self.grid[0][0] = (0, 0, 0, 0, 1.0)
        self.used_patches.add(0)
        for r in range(15):
            for c in range(15):
                if r == 0 and c == 0: continue
                l = np.rot90(self.patches[self.grid[r][c-1][0]], self.grid[r][c-1][1]) if (c>0 and self.grid[r][c-1]) else None
                t = np.rot90(self.patches[self.grid[r-1][c][0]], self.grid[r-1][c][1]) if (r>0 and self.grid[r-1][c]) else None
                match = self._find_best_match(l, t)
                if match:
                    self.grid[r][c] = match
                    self.used_patches.add(match[0])
                    print(f"NCC Matched ({r},{c}): Patch {match[0]}, Score: {match[4]:.4f}")
                else: print(f"WARNING: No match for ({r},{c})")

    def assemble(self):
        coords = [[(0, 0) for _ in range(15)] for _ in range(15)]
        for r in range(15):
            for c in range(1, 15):
                if self.grid[r][c] and self.grid[r][c-1]: coords[r][c] = (coords[r][c-1][0] + (128-self.grid[r][c][2]), coords[r][c][1])
                elif self.grid[r][c]: coords[r][c] = (coords[r][c-1][0] + 96, coords[r][c][1])
        for c in range(15):
            for r in range(1, 15):
                if self.grid[r][c] and self.grid[r-1][c]: coords[r][c] = (coords[r][c][0], coords[r-1][c][1] + (128-self.grid[r][c][3]))
                elif self.grid[r][c]: coords[r][c] = (coords[r][c][0], coords[r-1][c][1] + 96)
        
        mx = max(coords[r][c][0]+128 for r in range(15) for c in range(15) if self.grid[r][c])
        my = max(coords[r][c][1]+128 for r in range(15) for c in range(15) if self.grid[r][c])
        canvas = np.zeros((my, mx, 3), dtype=np.uint8)
        for r in range(15):
            for c in range(15):
                if self.grid[r][c]:
                    px, py = coords[r][c]
                    canvas[py:py+128, px:px+128] = np.rot90(self.patches[self.grid[r][c][0]], self.grid[r][c][1])
        return canvas

if __name__ == "__main__":
    ms = MapStitcherNCC("patches")
    ms.stitch()
    cv2.imwrite("reconstructed_ncc.png", ms.assemble())
