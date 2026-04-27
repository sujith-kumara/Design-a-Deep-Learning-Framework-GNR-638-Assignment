
import os
import cv2
import numpy as np
from collections import deque

class ProductionMapStitcher:
    def __init__(self, patch_dir, threshold_high=0.92, threshold_low=0.85):
        self.patch_dir = patch_dir
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.patches = self._load_patches()
        
        # global results: id -> (py, px, rot)
        self.placed_positions = {} 
        self.unplaced = set(self.patches.keys())
        self.queue = deque()

    def _load_patches(self):
        p = {}
        for f in sorted(os.listdir(self.patch_dir)):
            if f.endswith('.png'):
                try:
                    id_val = int(''.join(filter(str.isdigit, f)))
                    img = cv2.imread(os.path.join(self.patch_dir, f))
                    if img is not None: 
                        p[id_val] = img.astype(np.float32) / 255.0
                except: continue
        return p

    def _get_score(self, template, candidate):
        if template.shape[0] < 5 or template.shape[1] < 5: return 0.0, (0, 0)
        res = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc

    def _get_consensus(self, py, px, img, threshold):
        """
        Returns (avg_score, offsets_consistent)
        Checks all existing neighbors for pixel-level agreement.
        """
        scores = []
        agreements = []
        for opid, (opy, opx, orot) in self.placed_positions.items():
            dy, dx = py - opy, px - opx
            if abs(dy) > 130 or abs(dx) > 130: continue
            
            oimg = np.rot90(self.patches[opid], orot)
            y_start, y_end = max(py, opy), min(py + 128, opy + 128)
            x_start, x_end = max(px, opx), min(px + 128, opx + 128)
            
            if y_start < y_end and x_start < x_end:
                self_part = img[int(y_start-py):int(y_end-py), int(x_start-px):int(x_end-px)]
                other_part = oimg[int(y_start-opy):int(y_end-opy), int(x_start-opx):int(x_end-opx)]
                score, _ = self._get_score(self_part, other_part)
                scores.append(score)
        
        if not scores: return 0.0
        avg_score = sum(scores) / len(scores)
        return avg_score

    def run(self):
        # Anchor at P0
        self.placed_positions[0] = (0.0, 0.0, 0)
        self.unplaced.remove(0)
        self.queue.append(0)
        
        # PASS 1: Backbone (High Threshold)
        print(f"PASS 1: Building Map Backbone (Threshold: {self.threshold_high})")
        self._execution_loop(self.threshold_high)
        
        # PASS 2: Fills (Lower Threshold)
        if self.unplaced:
            print(f"PASS 2: Filling Ambiguous Gaps (Threshold: {self.threshold_low})")
            # Re-add all placed to queue to check neighbors for unplaced
            self.queue.extend(list(self.placed_positions.keys()))
            self._execution_loop(self.threshold_low)

    def _execution_loop(self, current_threshold):
        directions = ['right', 'left', 'bottom', 'top']
        iters = 0
        while self.queue and self.unplaced and iters < 1000:
            anchor_id = self.queue.popleft()
            if anchor_id not in self.placed_positions: continue
            
            ay, ax, arot = self.placed_positions[anchor_id]
            anchor_img = np.rot90(self.patches[anchor_id], arot)
            
            for cand_id in list(self.unplaced):
                best_match = (-1, None)
                for rot in [0, 1, 2, 3]:
                    cand_img = np.rot90(self.patches[cand_id], rot)
                    for name in directions:
                        # Template match strips
                        if name == 'right': t, s = anchor_img[:, -32:], cand_img[:, :64]
                        elif name == 'left': t, s = anchor_img[:, :32], cand_img[:, -64:]
                        elif name == 'bottom': t, s = anchor_img[-32:, :], cand_img[:64, :]
                        else: t, s = anchor_img[:32, :], cand_img[-64:, :]
                        
                        score, loc = self._get_score(t, s)
                        if score > 0.5:
                            # Relative pixel math
                            if name == 'right': npx, npy = ax+96-loc[0], ay-loc[1]
                            elif name == 'left': npx, npy = ax-(64+loc[0]), ay-loc[1]
                            elif name == 'bottom': npy, npx = ay+96-loc[1], ax-loc[0]
                            else: npy, npx = ay-(64+loc[1]), ax-loc[0]
                            
                            c_score = self._get_consensus(npy, npx, cand_img, current_threshold)
                            if c_score > best_match[0]:
                                best_match = (c_score, (npy, npx, rot))
                
                if best_match[0] > current_threshold:
                    ny, nx, r = best_match[1]
                    self.placed_positions[cand_id] = (ny, nx, r)
                    self.unplaced.remove(cand_id)
                    self.queue.append(cand_id)
                    print(f"   [LOCKED] Patch {cand_id} at ({int(ny)}, {int(nx)}) Score: {best_match[0]:.4f}")
            iters += 1

    def assemble(self):
        if not self.placed_positions: return None
        
        # Coordinate Normalization
        coords = list(self.placed_positions.values())
        min_y = min(c[0] for c in coords)
        min_x = min(c[1] for c in coords)
        
        # Re-offset to 0,0
        normalized = {}
        for pid, (py, px, rot) in self.placed_positions.items():
            normalized[pid] = (py - min_y, px - min_x, rot)
            
        max_y = int(max(c[0] for c in normalized.values())) + 128
        max_x = int(max(c[1] for c in normalized.values())) + 128
        
        # Vectorized Feathering Assembly
        canvas = np.zeros((max_y, max_x, 3), dtype=np.float32)
        weight = np.zeros((max_y, max_x, 3), dtype=np.float32)
        
        # Create 128x128 linear feathering mask
        ramp = np.linspace(0, 1, 16)
        mask_1d = np.ones(128)
        mask_1d[:16] = ramp
        mask_1d[-16:] = ramp[::-1]
        mask_2d = np.outer(mask_1d, mask_1d)
        mask_3d = np.repeat(mask_2d[:, :, np.newaxis], 3, axis=2)
        
        for pid, (py, px, rot) in normalized.items():
            img = np.rot90(self.patches[pid], rot)
            y, x = int(py), int(px)
            canvas[y:y+128, x:x+128] += img * mask_3d
            weight[y:y+128, x:x+128] += mask_3d
            
        # Safe normalization (Handle division by zero)
        result = np.divide(canvas, weight, out=np.zeros_like(canvas), where=weight != 0)
        
        # Memory Guard: Clear patch cache
        self.patches.clear()
        
        return (result * 255).astype(np.uint8)

if __name__ == "__main__":
    stitcher = ProductionMapStitcher("patches")
    stitcher.run()
    map_final = stitcher.assemble()
    if map_final is not None:
        cv2.imwrite("production_stitched_map.png", map_final)
        print("Success: Production-grade map saved.")
