
import os
import cv2
import numpy as np
from collections import deque

class BFSFrontierStitcher:
    def __init__(self, patch_dir, threshold=0.8, patch_dim=128):
        self.patch_dir = patch_dir
        self.threshold = threshold
        self.patch_dim = patch_dim
        self.patches = self._load_patches()
        
        # id -> (py, px, rot) - coordinates in total pixels relative to patch_0
        self.placed_positions = {} 
        self.unplaced = set(self.patches.keys())
        self.queue = deque()

    def _load_patches(self):
        p = {}
        for f in os.listdir(self.patch_dir):
            if f.endswith('.png'):
                id_val = int(''.join(filter(str.isdigit, f)))
                img = cv2.imread(os.path.join(self.patch_dir, f))
                if img is not None: p[id_val] = img
        return p

    def _get_score(self, template, candidate):
        if template.shape[0] < 5 or template.shape[1] < 5: return 0
        if candidate.shape[0] < template.shape[0] or candidate.shape[1] < template.shape[1]:
            return 0
        res = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val

    def _match_patches(self, anchor_img, cand_img, direction):
        if direction == 'right':
            template = anchor_img[:, -32:]
            search_area = cand_img[:, :64]
        elif direction == 'left':
            template = anchor_img[:, :32]
            search_area = cand_img[:, -64:]
        elif direction == 'bottom':
            template = anchor_img[-32:, :]
            search_area = cand_img[:64, :]
        else: # top
            template = anchor_img[:32, :]
            search_area = cand_img[-64:, :]
            
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc

    def _validate_geometry(self, py, px, img):
        for opid, (opy, opx, orot) in self.placed_positions.items():
            dy, dx = py - opy, px - opx
            if abs(dy) > 130 or abs(dx) > 130: continue
            if abs(dy) < 1 and abs(dx) < 1: continue 
            
            oimg = np.rot90(self.patches[opid], orot)
            y_start, y_end = max(py, opy), min(py + 128, opy + 128)
            x_start, x_end = max(px, opx), min(px + 128, opx + 128)
            
            if y_start < y_end and x_start < x_end:
                self_part = img[int(y_start-py):int(y_end-py), int(x_start-px):int(x_end-px)]
                other_part = oimg[int(y_start-opy):int(y_end-opy), int(x_start-opx):int(x_end-opx)]
                if self._get_score(self_part, other_part) < self.threshold: return False
        return True

    def run(self):
        self.placed_positions[0] = (0, 0, 0)
        self.unplaced.remove(0)
        self.queue.append(0)
        
        print("Starting BFS Discovery...")
        
        while self.queue and self.unplaced:
            anchor_id = self.queue.popleft()
            ay, ax, arot = self.placed_positions[anchor_id]
            anchor_img = np.rot90(self.patches[anchor_id], arot)
            
            directions = ['right', 'left', 'bottom', 'top']
            
            for cand_id in list(self.unplaced):
                best_for_cand = (-1, None)
                for rot in [0, 1, 2, 3]:
                    cand_img = np.rot90(self.patches[cand_id], rot)
                    for name in directions:
                        score, loc = self._match_patches(anchor_img, cand_img, name)
                        if score > self.threshold:
                            if name == 'right': npx, npy = ax+96-loc[0], ay-loc[1]
                            elif name == 'left': npx, npy = ax-(64+loc[0]), ay-loc[1]
                            elif name == 'bottom': npy, npx = ay+96-loc[1], ax-loc[0]
                            else: npy, npx = ay-(64+loc[1]), ax-loc[0]
                            
                            if score > best_for_cand[0]:
                                best_for_cand = (score, (npy, npx, rot))
                
                if best_for_cand[0] > self.threshold:
                    ny, nx, r = best_for_cand[1]
                    if self._validate_geometry(ny, nx, np.rot90(self.patches[cand_id], r)):
                        self.placed_positions[cand_id] = (ny, nx, r)
                        self.unplaced.remove(cand_id)
                        self.queue.append(cand_id)
                        print(f"   [MATCH] Patch {cand_id} placed at Pixel ({int(ny)}, {int(nx)}). Score: {best_for_cand[0]:.4f}")

    def assemble(self):
        if not self.placed_positions: return None
        p_coords = list(self.placed_positions.values())
        
        # User Constraint: patch_0 is (0,0) Top-Left. 
        # So we clip everything that resulted in negative coordinates relative to it.
        min_y = 0 
        min_x = 0
        max_y = int(max(c[0] for c in p_coords))
        max_x = int(max(c[1] for c in p_coords))
        
        canvas = np.zeros((max_y + 128, max_x + 128, 3), dtype=np.uint8)
        
        for pid, (py, px, rot) in self.placed_positions.items():
            if py < 0 or px < 0: continue # Following "patch_0 is top-left" rule
            img = np.rot90(self.patches[pid], rot)
            canvas[int(py):int(py)+128, int(px):int(px)+128] = img
            
        return canvas

if __name__ == "__main__":
    stitcher = BFSFrontierStitcher("patches", threshold=0.8)
    stitcher.run()
    result = stitcher.assemble()
    if result is not None:
        cv2.imwrite("reconstructed_bfs_anchored.png", result)
        print("Success: Final BFS reconstructed map (Anchored at P0) saved.")
