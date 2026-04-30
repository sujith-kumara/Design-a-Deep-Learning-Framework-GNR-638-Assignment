import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import easyocr
from collections import deque
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

# ===========================================================================
# 1. GPU ACCELERATION SETUP (CuPy Fallback)
# ===========================================================================
try:
    import cupy as cp
    _test = cp.zeros(1)
    del _test
    GPU_AVAILABLE = True
    xp = cp
    print("[GPU] CuPy detected — using GPU-accelerated scoring.")
except Exception:
    import numpy as cp
    GPU_AVAILABLE = False
    xp = np
    print("[GPU] CuPy not available — falling back to CPU.")

# ===========================================================================
# 2. MAP STITCHING MODULE
# ===========================================================================
def _to_gpu(arr):
    return cp.asarray(arr) if GPU_AVAILABLE else arr

def _ncc_batch(A, B):
    N = A.shape[0]
    A_flat = A.reshape(N, -1)
    B_flat = B.reshape(N, -1)
    A_mu = A_flat.mean(axis=1, keepdims=True)
    B_mu = B_flat.mean(axis=1, keepdims=True)
    A_c = A_flat - A_mu
    B_c = B_flat - B_mu
    num = (A_c * B_c).sum(axis=1)
    denom = xp.sqrt((A_c ** 2).sum(axis=1) * (B_c ** 2).sum(axis=1))
    ncc = xp.where(denom < 1e-6, xp.zeros(N, dtype=xp.float32), num / denom)
    return ncc.astype(xp.float32)

def _mse_batch(A, B):
    N = A.shape[0]
    diff = (A - B).reshape(N, -1)
    mse = (diff ** 2).mean(axis=1)
    return (1.0 - mse / 65025.0).astype(xp.float32)

def _combined_batch(A, B):
    TEXTURE_LOW  = 8.0
    TEXTURE_HIGH = 30.0
    N = A.shape[0]
    std = A.reshape(N, -1).std(axis=1)
    t = xp.clip((std - TEXTURE_LOW) / (TEXTURE_HIGH - TEXTURE_LOW), 0.0, 1.0)
    ncc = _ncc_batch(A, B)
    mse = _mse_batch(A, B)
    ncc_01 = (ncc + 1.0) / 2.0
    return (t * ncc_01 + (1.0 - t) * mse).astype(xp.float32)

def _extract_strip(patch, side, strip_w):
    h, w = patch.shape[:2]
    s = min(strip_w, w if side in ("right_edge", "left_edge") else h)
    if side == "right_edge":  return patch[:, w - s:]
    if side == "left_edge":   return patch[:, :s]
    if side == "bottom_edge": return patch[h - s:, :]
    if side == "top_edge":    return patch[:s, :]
    raise ValueError(f"Unknown side: {side}")

def _is_strip_mostly_black(strip, black_thresh=15, black_frac=0.85):
    gray = strip.mean(axis=2) if strip.ndim == 3 else strip
    return float((gray < black_thresh).mean()) >= black_frac

def _detect_strip_w(ph, pw):
    return max(8, min(64, int(round(min(ph, pw) * 0.10))))

def _detect_global_overlap(patches, max_overlap=64):
    if len(patches) < 2: return 0
    all_ids = sorted(patches.keys())
    ph, pw  = patches[all_ids[0]].shape[:2]
    max_ov  = min(max_overlap, pw // 4, ph // 4)
    if max_ov < 1: return 0

    n_anchors  = min(8, len(all_ids))
    step       = max(1, len(all_ids) // n_anchors)
    anchor_ids = all_ids[::step][:n_anchors]
    other_imgs = [patches[i].astype(np.float32) for i in all_ids]
    votes      = np.zeros(max_ov + 1, dtype=np.int32)

    for aid in anchor_ids:
        af         = patches[aid].astype(np.float32)
        right_full = xp.asarray(af[:, pw - max_ov:])
        bot_full   = xp.asarray(af[ph - max_ov:, :])
        others_left = xp.asarray(np.stack([o[:, :max_ov] for o in other_imgs]))
        others_top  = xp.asarray(np.stack([o[:max_ov, :] for o in other_imgs]))

        best_h = np.full(max_ov + 1, np.inf)
        best_v = np.full(max_ov + 1, np.inf)

        for ov in range(1, max_ov + 1):
            nb_h = right_full[:, max_ov - ov:]
            ca_h = others_left[:, :, :ov]
            nb_h = xp.broadcast_to(nb_h[xp.newaxis], ca_h.shape)
            mse_h = ((nb_h - ca_h) ** 2).reshape(len(other_imgs), -1).mean(axis=1)

            nb_v = bot_full[max_ov - ov:]
            ca_v = others_top[:, :ov]
            nb_v = xp.broadcast_to(nb_v[xp.newaxis], ca_v.shape)
            mse_v = ((nb_v - ca_v) ** 2).reshape(len(other_imgs), -1).mean(axis=1)

            if GPU_AVAILABLE:
                mse_h = cp.asnumpy(mse_h)
                mse_v = cp.asnumpy(mse_v)

            aid_pos = all_ids.index(aid)
            mse_h[aid_pos] = np.inf
            mse_v[aid_pos] = np.inf
            best_h[ov] = float(mse_h.min())
            best_v[ov] = float(mse_v.min())

        both = np.minimum(best_h, best_v)
        both[0] = np.inf
        votes[int(np.argmin(both))] += 1

    best_ov = int(np.argmax(votes))
    if best_ov == 0: return 0

    strip  = min(16, pw, ph)
    af0    = patches[anchor_ids[0]].astype(np.float32)
    others_np = [patches[i].astype(np.float32) for i in all_ids if i != anchor_ids[0]]
    base = min(
        min(float(np.mean((af0[:, pw-strip:] - o[:, :strip])**2)) for o in others_np),
        min(float(np.mean((af0[ph-strip:, :] - o[:strip, :])**2)) for o in others_np),
    )
    af0g  = xp.asarray(af0)
    right = af0g[:, pw - best_ov:]
    bot   = af0g[ph - best_ov:, :]
    winner_mse = min(
        float(min(float(xp.mean((right - xp.asarray(o[:, :best_ov]))**2)) for o in others_np)),
        float(min(float(xp.mean((bot   - xp.asarray(o[:best_ov, :])**2))) for o in others_np)),
    )
    if winner_mse > base * 0.40:
        return 0

    return best_ov

def _score_all_candidates_gpu(unplaced_imgs, neighbours, strip_w):
    N = len(unplaced_imgs)
    scores    = np.zeros(N, dtype=np.float32)
    n_borders = np.zeros(N, dtype=np.int32)

    edge_map = {
        "left":  ("right_edge",  "left_edge"),
        "right": ("left_edge",   "right_edge"),
        "up":    ("bottom_edge", "top_edge"),
        "down":  ("top_edge",    "bottom_edge"),
    }

    for direction, nb_img in neighbours.items():
        nb_side, cand_side = edge_map[direction]
        nb_strip = _extract_strip(nb_img, nb_side, strip_w)
        if _is_strip_mostly_black(nb_strip): continue

        batch_A, batch_B, valid_idx = [], [], []
        for i, cand_img in enumerate(unplaced_imgs):
            cand_strip = _extract_strip(cand_img, cand_side, strip_w)
            if nb_strip.shape != cand_strip.shape: continue
            batch_A.append(nb_strip)
            batch_B.append(cand_strip)
            valid_idx.append(i)

        if not batch_A: continue

        A_gpu = _to_gpu(np.stack(batch_A).astype(np.float32))
        B_gpu = _to_gpu(np.stack(batch_B).astype(np.float32))
        edge_scores = _combined_batch(A_gpu, B_gpu)
        if GPU_AVAILABLE: edge_scores = cp.asnumpy(edge_scores)

        for j, i in enumerate(valid_idx):
            scores[i]    += float(edge_scores[j])
            n_borders[i] += 1

    return np.where(n_borders > 0, scores / np.maximum(n_borders, 1), -np.inf)

class ProductionMapStitcher:
    def __init__(self, patch_dir, score_threshold=0.3):
        self.patch_dir       = patch_dir
        self.score_threshold = score_threshold
        self._patches        = {}   
        self._grid           = {}   
        self._pixels         = {}   
        self._patch_h        = 0
        self._patch_w        = 0
        self._global_overlap = 0
        self._strip_w        = 0

        print("[Stitcher] Loading patches from:", patch_dir)
        self._patches = self._load_patches()
        if self._patches:
            sample = next(iter(self._patches.values()))
            self._patch_h, self._patch_w = sample.shape[:2]

    def _load_patches(self):
        patches = {}
        if not os.path.exists(self.patch_dir): return patches
        for f in sorted(os.listdir(self.patch_dir)):
            if f.endswith('.png'):
                try:
                    id_val = int(''.join(filter(str.isdigit, f)))
                    img = cv2.imread(os.path.join(self.patch_dir, f))
                    if img is not None: patches[id_val] = img
                except Exception: continue
        print(f"[Stitcher] Loaded {len(patches)} patches.")
        return patches

    def _grid_neighbours(self):
        candidates = set()
        for (r, c) in self._grid:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in self._grid and nr >= 0 and nc >= 0:
                    candidates.add((nr, nc))
        return list(candidates)

    def _get_neighbour_images(self, row, col):
        result = {}
        for direction, (dr, dc) in {'left':(0,-1),'right':(0,1),'up':(-1,0),'down':(1,0)}.items():
            pos = (row + dr, col + dc)
            if pos in self._grid: result[direction] = self._grid[pos]
        return result

    def _get_neighbour_positions(self, row, col):
        result = {}
        for direction, (dr, dc) in {'left':(0,-1),'right':(0,1),'up':(-1,0),'down':(1,0)}.items():
            pos = (row + dr, col + dc)
            if pos in self._grid: result[direction] = pos
        return result

    def _place(self, row, col, img, pixel_origin):
        self._grid[(row, col)]   = img
        self._pixels[(row, col)] = pixel_origin

    def _place_relative(self, row, col, img, neighbour_pos, direction, overlap):
        ny, nx = self._pixels[neighbour_pos]
        nh, nw = self._grid[neighbour_pos].shape[:2]
        ph, pw = img.shape[:2]
        if   direction == 'left':  y, x = ny,              nx + nw - overlap
        elif direction == 'up':    y, x = ny + nh - overlap, nx
        elif direction == 'right': y, x = ny,              nx - pw + overlap
        elif direction == 'down':  y, x = ny - ph + overlap, nx
        else: raise ValueError(f"Unknown direction: {direction}")
        self._place(row, col, img, (y, x))

    def _refine_overlap(self, placed_img, new_img, direction):
        sw2 = self._strip_w * 2
        go  = self._global_overlap

        if direction == 'left':
            nb_edge, cand_edge = placed_img[:, -sw2:].astype(np.float32), new_img[:, :sw2].astype(np.float32)
            def mse_at(ov): return float(np.mean((nb_edge[:, -ov:] - cand_edge[:, :ov]) ** 2))
        elif direction == 'right':
            nb_edge, cand_edge = placed_img[:, :sw2].astype(np.float32), new_img[:, -sw2:].astype(np.float32)
            def mse_at(ov): return float(np.mean((nb_edge[:, :ov] - cand_edge[:, -ov:]) ** 2))
        elif direction == 'up':
            nb_edge, cand_edge = placed_img[-sw2:, :].astype(np.float32), new_img[:sw2, :].astype(np.float32)
            def mse_at(ov): return float(np.mean((nb_edge[-ov:, :] - cand_edge[:ov, :]) ** 2))
        else:
            nb_edge, cand_edge = placed_img[:sw2, :].astype(np.float32), new_img[-sw2:, :].astype(np.float32)
            def mse_at(ov): return float(np.mean((nb_edge[:ov, :] - cand_edge[-ov:, :]) ** 2))

        lo = max(1, go - 8) if go > 0 else 1
        hi = min(sw2, go + 8) if go > 0 else sw2

        best_mse, best_ov = np.inf, go if go > 0 else self._strip_w
        for ov in range(lo, hi + 1):
            m = mse_at(ov)
            if m < best_mse: best_mse, best_ov = m, ov
        return best_ov

    def run(self):
        if 0 not in self._patches: return
        ph, pw = self._patch_h, self._patch_w
        self._global_overlap = _detect_global_overlap(self._patches)
        self._strip_w = self._global_overlap if self._global_overlap > 0 else _detect_strip_w(ph, pw)

        anchor = self._patches[0]
        self._place(0, 0, anchor, (0, 0))

        unplaced = {k: v for k, v in self._patches.items() if k != 0}
        max_iters = len(self._patches) ** 2
        itr = 0

        while unplaced and itr < max_iters:
            itr += 1
            best_score, best_pid, best_pos = -np.inf, None, None
            empty_cells = self._grid_neighbours()

            unplaced_pids = list(unplaced.keys())
            unplaced_imgs = [unplaced[p] for p in unplaced_pids]

            clean_cells, black_cells = [], []
            for pos in empty_cells:
                r, c = pos
                neighbours = self._get_neighbour_images(r, c)
                if not neighbours: continue
                all_black = all(_is_strip_mostly_black(_extract_strip(nb_img, {"left":"right_edge","right":"left_edge","up":"bottom_edge","down":"top_edge"}[d], self._strip_w)) for d, nb_img in neighbours.items())
                (black_cells if all_black else clean_cells).append(pos)

            for pos in (clean_cells or black_cells):
                neighbours = self._get_neighbour_images(*pos)
                avg_scores = _score_all_candidates_gpu(unplaced_imgs, neighbours, self._strip_w)
                best_i     = int(np.argmax(avg_scores))
                pos_score  = float(avg_scores[best_i])
                if pos_score > best_score:
                    best_score = pos_score
                    best_pid   = unplaced_pids[best_i]
                    best_pos   = pos

            if best_pid is None or (best_score < self.score_threshold and best_score <= -0.5): break

            row, col   = best_pos
            placed_img = unplaced[best_pid]          
            nb_positions = self._get_neighbour_positions(row, col)
            nb_images    = self._get_neighbour_images(row, col)
            ref_dir, ref_nb_pos = next(iter(nb_positions.items()))
            
            actual_overlap = self._refine_overlap(nb_images[ref_dir], placed_img, ref_dir)
            self._place_relative(row, col, placed_img, neighbour_pos=ref_nb_pos, direction=ref_dir, overlap=actual_overlap)
            del unplaced[best_pid]

    def assemble(self):
        if not self._grid: return None
        min_y = min(y for y, x in self._pixels.values())
        min_x = min(x for y, x in self._pixels.values())
        if min_y < 0 or min_x < 0:
            self._pixels = {k: (y - min_y, x - min_x) for k, (y, x) in self._pixels.items()}

        max_y = max(self._pixels[k][0] + self._grid[k].shape[0] for k in self._grid)
        max_x = max(self._pixels[k][1] + self._grid[k].shape[1] for k in self._grid)

        canvas  = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        written = np.zeros((max_y, max_x),    dtype=bool)

        ordered = sorted(self._grid.keys(), key=lambda k: (k[0] + k[1], k[0]))
        for key in ordered:
            img, (y, x) = self._grid[key], self._pixels[key]
            h, w = img.shape[:2]
            mask3 = ~written[y:y+h, x:x+w, np.newaxis]
            canvas[y:y+h, x:x+w] = np.where(mask3, img, canvas[y:y+h, x:x+w])
            written[y:y+h, x:x+w] = True
        return canvas

# ===========================================================================
# 3. VQA & OCR INFERENCE MODULE
# ===========================================================================
MODEL_NAME = "./llava-1.6-mistral-offline" 
DEVICE     = "cuda"

def load_models():
    print(f"Loading LLaVA-NeXT: {MODEL_NAME} (8-bit mode) | device={DEVICE.upper()}")
    processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    llava_model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto" 
    )
    llava_model.eval()
    
    print("Loading EasyOCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))
    return processor, llava_model, ocr_reader

def extract_map_text(ocr_reader, img_bgr):
    """Takes a raw BGR numpy array directly from the stitcher in memory."""
    print("[OCR] Upscaling stitched map for clarity...")
    img_h, img_w = img_bgr.shape[:2] 
    
    scale_factor = 2
    width = int(img_w * scale_factor)
    height = int(img_h * scale_factor)
    upscaled_img = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2GRAY)
    
    raw_ocr = ocr_reader.readtext(
        gray_img, 
        mag_ratio=1.5,         
        text_threshold=0.4,    
        link_threshold=0.3,    
        canvas_size=2000       
    )
    
    ocr_data = []
    for bbox, text, prob in raw_ocr:
        if prob > 0.35 and len(text.strip()) > 2: 
            xs = [pt[0] / scale_factor for pt in bbox]
            ys = [pt[1] / scale_factor for pt in bbox]
            
            # Standard Cartesian: (0,0) is bottom-left
            cx = int(sum(xs) / 4)
            cy = int(img_h - (sum(ys) / 4))
            
            # Determine Grid Quadrant
            ns = "North" if cy > (img_h / 2) else "South"
            ew = "East" if cx > (img_w / 2) else "West"
            quadrant = f"{ns}-{ew}"
            
            point_str = f"(X:{cx}, Y:{cy}) [{quadrant}]"
            ocr_data.append({"text": text.lower().strip(), "box": point_str})
            
    print(f"[OCR] Extracted {len(ocr_data)} clean text elements.\n")
    return ocr_data

def get_relevant_ocr(question: str, options: list, ocr_data: list) -> str:
    query_str = f"{question} {' '.join([str(o) for o in options if pd.notna(o)])}".lower()
    relevant_ocr = []
    for item in ocr_data:
        text = item['text']
        sig_words = [w for w in text.split() if len(w) >= 4]
        if text in query_str or any(w in query_str for w in sig_words):
            relevant_ocr.append(f"'{text}' is located at {item['box']}")
            
    if not relevant_ocr:
        return "No explicit coordinates found."
    return "\n".join(list(set(relevant_ocr))[:15])

def answer_question(processor, model, image: Image.Image, question: str, options: list, ocr_context: str) -> str:
    valid = [(str(i+1), opt) for i, opt in enumerate(options) if pd.notna(opt) and str(opt).strip() != ""]
    options_text = "\n".join([f"{num}) {opt}" for num, opt in valid])
    
    prompt_text = (
        f"You are a spatial reasoning AI analyzing an OpenStreetMap.\n"
        f"Coordinate System: Standard Cartesian (X, Y). (0,0) is BOTTOM-LEFT.\n"
        f"- NORTH = LARGER Y value. SOUTH = SMALLER Y value.\n"
        f"- EAST = LARGER X value. WEST = SMALLER X value.\n\n"
        f"Context Data (Coordinates & Quadrants):\n{ocr_context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Instructions:\n"
        f"1. Use the Context Data to find the locations of the landmarks.\n"
        f"2. To find relative directions, simply compare the X and Y values.\n"
        f"3. If a landmark is missing from the Context Data, look at the map image directly.\n"
        f"Conclude your response with the exact phrase: 'Final Answer: [Number]' where [Number] is 1, 2, 3, 4, or 5."
    )
    
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

    output = model.generate(**inputs, max_new_tokens=250, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
    generated_text = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    match = re.search(r'Final Answer:\s*([1-5])', generated_text, re.IGNORECASE)
    pred_num = match.group(1) if match else (re.findall(r'\b([1-5])\b', generated_text)[-1] if re.findall(r'\b([1-5])\b', generated_text) else "5")
    
    return pred_num

def run_pipeline(test_dir: str, output_path: str):
    # 1. Stitch Map
    patch_dir = os.path.join(test_dir, "patches") # <-- Point to the subfolder
    print(f"--- 1. Assembling map patches from {patch_dir} ---")
    
    stitcher = ProductionMapStitcher(patch_dir) # <-- Pass patch_dir instead of test_dir
    stitcher.run()
    canvas_bgr = stitcher.assemble()
    
    if canvas_bgr is None:
        print("[CRITICAL ERROR] Stitcher failed to assemble map. Exiting.")
        return
    
    # 2. Convert BGR to PIL RGB Image for LLaVA
    image_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    map_pil_image = Image.fromarray(image_rgb)
    
    # 3. Load VQA Logic
    print(f"\n--- 2. Initializing VQA Models ---")
    processor, llava_model, ocr_reader = load_models()
    
    # Extract OCR passing the BGR array directly
    global_ocr_data = extract_map_text(ocr_reader, canvas_bgr)
    
    # Load test.csv from the ROOT of test_dir
    csv_path = os.path.join(test_dir, 'test.csv') # <-- Stays at root
    if not os.path.exists(csv_path):
        print(f"[CRITICAL ERROR] Could not find {csv_path}. Exiting.")
        return
    df = pd.read_csv(csv_path)
    
    # 4. Generate Predictions
    print(f"\n--- 3. Running Inference ---")
    results = []
    for idx, row in df.iterrows():
        # UPDATE 1: Match the exact column names 'option_1', 'option_2', etc.
        options = [row.get(f"option_{i}") for i in range(1, 5)]
        options.append("Information not present on the map.")
        
        ocr_context = get_relevant_ocr(row["question"], options, global_ocr_data)
        pred_num = answer_question(processor, llava_model, map_pil_image, row["question"], options, ocr_context)
            
        print(f"[{idx+1:02d}] id={row['id']} pred={pred_num}")
        
        # UPDATE 2: Match the exact output format required by sample_submission.csv
        results.append({
            "id": row["id"], 
            "question_num": row["id"], # 'question_num' matches 'id' in your sample
            "option": pred_num
        })
        
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nPredictions saved successfully to {output_path}")

# ===========================================================================
# 4. CLI ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNR Map Stitcher & VQA Inference Pipeline")
    parser.add_argument("--test_dir", type=str, required=True, help="Absolute path to directory containing patches/ and test.csv")
    args = parser.parse_args()
    
    out_file = "submission.csv"
    run_pipeline(args.test_dir, out_file)