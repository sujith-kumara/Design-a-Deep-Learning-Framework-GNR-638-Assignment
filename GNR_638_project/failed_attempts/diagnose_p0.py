
import cv2
import os
import numpy as np

def get_score(template, candidate):
    res = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val

patch_dir = "patches"
p0 = cv2.imread(os.path.join(patch_dir, "patch_0.png"))
best_overall = []

for f in os.listdir(patch_dir):
    if f == "patch_0.png" or not f.endswith(".png"): continue
    p_cand = cv2.imread(os.path.join(patch_dir, f))
    for rot in [0, 1, 2, 3]:
        img = np.rot90(p_cand, rot)
        
        # Right check
        s_r = get_score(p0[:, -32:], img[:, :64])
        # Bottom check
        s_b = get_score(p0[-32:, :], img[:64, :])
        
        mx = max(s_r, s_b)
        if mx > 0.3:
            best_overall.append((f, rot, mx))

best_overall.sort(key=lambda x: x[2], reverse=True)
for item in best_overall[:10]:
    print(f"Cand {item[0]} Rot {item[1]} Score {item[2]:.4f}")
