import os
import cv2
import numpy as np
import torch
from collections import deque
import pandas as pd
from PIL import Image
import re
import argparse
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

# ── Config ──────────────────────────────────────────────────────────────────
# 1. Model is read locally from your current directory
MODEL_NAME     = "./llava-v1.6-mistral-7b-hf" # <-- Ensure this matches your local model folder name
DEVICE         = "cuda"   

# 2. OUTPUTS: Saved directly to the current working directory
MAP_IMAGE_PATH = "production_stitched_map.png" 
OUTPUT_PATH    = "submission.csv"
# ────────────────────────────────────────────────────────────────────────────

class ProductionMapStitcher:
    def __init__(self, patch_dir, threshold_high=0.92, threshold_low=0.85, 
                 overlap_ratio=0.25, search_ratio=0.5):
        self.patch_dir = patch_dir
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        
        self.overlap_ratio = overlap_ratio
        self.search_ratio = search_ratio
        
        self.patches = self._load_patches()
        
        self.use_cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
        if self.use_cv2_cuda:
            print("INFO: OpenCV CUDA detected. Hardware-accelerated matching enabled.")
        else:
            print("INFO: OpenCV CUDA not found. Matching on CPU (Assembly will still use PyTorch GPU).")

        self.placed_positions = {} 
        self.unplaced = set(self.patches.keys())
        self.queue = deque()

    def _load_patches(self):
        p = {}
        for f in sorted(os.listdir(self.patch_dir)):
            if f.endswith('.png') or f.endswith('.jpg'):
                try:
                    id_val = int(''.join(filter(str.isdigit, f)))
                    img = cv2.imread(os.path.join(self.patch_dir, f))
                    if img is not None: 
                        p[id_val] = img.astype(np.float32) / 255.0
                except: continue
        return p

    def _get_score(self, template, candidate):
        if template.shape[0] < 5 or template.shape[1] < 5: return 0.0, (0, 0)
        
        if self.use_cv2_cuda:
            try:
                cand_gpu = cv2.cuda_GpuMat()
                cand_gpu.upload(candidate)
                temp_gpu = cv2.cuda_GpuMat()
                temp_gpu.upload(template)
                matcher = cv2.cuda.createTemplateMatching(cv2.CV_32F, cv2.TM_CCOEFF_NORMED)
                res_gpu = matcher.match(cand_gpu, temp_gpu)
                res = res_gpu.download()
            except cv2.error:
                res = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
        else:
            res = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
            
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc

    def _get_consensus(self, py, px, img, threshold):
        scores, agreements = [], []
        ch, cw = img.shape[:2]
        
        for opid, (opy, opx, orot, oh, ow) in self.placed_positions.items():
            dy, dx = py - opy, px - opx
            if abs(dy) > max(ch, oh) or abs(dx) > max(cw, ow): continue
            
            oimg = np.rot90(self.patches[opid], orot)
            
            y_start, y_end = max(py, opy), min(py + ch, opy + oh)
            x_start, x_end = max(px, opx), min(px + cw, opx + ow)
            
            if y_start < y_end and x_start < x_end:
                self_part = img[int(y_start-py):int(y_end-py), int(x_start-px):int(x_end-px)]
                other_part = oimg[int(y_start-opy):int(y_end-opy), int(x_start-opx):int(x_end-opx)]
                score, _ = self._get_score(self_part, other_part)
                scores.append(score)
        
        if not scores: return 0.0
        return sum(scores) / len(scores)

    def run(self):
        if not self.unplaced: return
        
        if 0 in self.unplaced:
            anchor_id = 0
        else:
            anchor_id = list(self.unplaced)[0]
            
        ah, aw = self.patches[anchor_id].shape[:2]
        self.placed_positions[anchor_id] = (0.0, 0.0, 0, ah, aw)
        self.unplaced.remove(anchor_id)
        self.queue.append(anchor_id)
        
        print(f"PASS 1: Building Map Backbone (Threshold: {self.threshold_high})")
        self._execution_loop(self.threshold_high)
        
        if self.unplaced:
            print(f"PASS 2: Filling Ambiguous Gaps (Threshold: {self.threshold_low})")
            self.queue.extend(list(self.placed_positions.keys()))
            self._execution_loop(self.threshold_low)

    def _execution_loop(self, current_threshold):
        directions = ['right', 'left', 'bottom', 'top']
        iters = 0
        while self.queue and self.unplaced and iters < 1000:
            anchor_id = self.queue.popleft()
            if anchor_id not in self.placed_positions: continue
            
            ay, ax, arot, ah, aw = self.placed_positions[anchor_id]
            anchor_img = np.rot90(self.patches[anchor_id], arot)
            
            ov_ah, ov_aw = int(ah * self.overlap_ratio), int(aw * self.overlap_ratio)
            
            for cand_id in list(self.unplaced):
                best_match = (-1, None)
                for rot in [0, 1, 2, 3]:
                    cand_img = np.rot90(self.patches[cand_id], rot)
                    ch, cw = cand_img.shape[:2]
                    
                    sr_ch, sr_cw = int(ch * self.search_ratio), int(cw * self.search_ratio)
                    
                    for name in directions:
                        if name == 'right':   t, s = anchor_img[:, -ov_aw:], cand_img[:, :sr_cw]
                        elif name == 'left':  t, s = anchor_img[:, :ov_aw], cand_img[:, -sr_cw:]
                        elif name == 'bottom': t, s = anchor_img[-ov_ah:, :], cand_img[:sr_ch, :]
                        else:                  t, s = anchor_img[:ov_ah, :], cand_img[-sr_ch:, :]
                        
                        score, loc = self._get_score(t, s)
                        if score > 0.5:
                            if name == 'right':   npx, npy = ax + (aw - ov_aw) - loc[0], ay - loc[1]
                            elif name == 'left':  npx, npy = ax - (cw - sr_cw) - loc[0], ay - loc[1]
                            elif name == 'bottom': npy, npx = ay + (ah - ov_ah) - loc[1], ax - loc[0]
                            else:                  npy, npx = ay - (ch - sr_ch) - loc[1], ax - loc[0]
                            
                            c_score = self._get_consensus(npy, npx, cand_img, current_threshold)
                            if c_score > best_match[0]:
                                best_match = (c_score, (npy, npx, rot, ch, cw))
                
                if best_match[0] > current_threshold:
                    ny, nx, r, n_ch, n_cw = best_match[1]
                    self.placed_positions[cand_id] = (ny, nx, r, n_ch, n_cw)
                    self.unplaced.remove(cand_id)
                    self.queue.append(cand_id)
                    print(f"   [LOCKED] Patch {cand_id} at ({int(ny)}, {int(nx)}) Score: {best_match[0]:.4f}")
            iters += 1

    def _get_feather_mask_torch(self, h, w, device, ramp_size=16):
        ramp_y = torch.linspace(0, 1, min(ramp_size, h//2), device=device)
        mask_y = torch.ones(h, device=device)
        mask_y[:len(ramp_y)] = ramp_y
        mask_y[-len(ramp_y):] = ramp_y.flip(0)

        ramp_x = torch.linspace(0, 1, min(ramp_size, w//2), device=device)
        mask_x = torch.ones(w, device=device)
        mask_x[:len(ramp_x)] = ramp_x
        mask_x[-len(ramp_x):] = ramp_x.flip(0)

        mask_2d = torch.outer(mask_y, mask_x)
        return mask_2d.unsqueeze(2).repeat(1, 1, 3)

    def assemble(self):
        if not self.placed_positions: return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"INFO: Assembling map on device: {device.type.upper()}")
        
        coords = list(self.placed_positions.values())
        min_y = min(c[0] for c in coords)
        min_x = min(c[1] for c in coords)
        
        normalized = {}
        for pid, (py, px, rot, h, w) in self.placed_positions.items():
            normalized[pid] = (py - min_y, px - min_x, rot, h, w)
            
        max_y = int(max(c[0] + c[3] for c in normalized.values()))
        max_x = int(max(c[1] + c[4] for c in normalized.values()))
        
        canvas = torch.zeros((max_y, max_x, 3), dtype=torch.float32, device=device)
        weight = torch.zeros((max_y, max_x, 3), dtype=torch.float32, device=device)
        
        for pid, (py, px, rot, h, w) in normalized.items():
            img_np = np.rot90(self.patches[pid], rot).copy()
            img_t = torch.from_numpy(img_np).to(device)
            mask_t = self._get_feather_mask_torch(h, w, device)
            
            y, x = int(py), int(px)
            canvas[y:y+h, x:x+w] += img_t * mask_t
            weight[y:y+h, x:x+w] += mask_t
            
        result = torch.where(weight != 0, canvas / weight, torch.zeros_like(canvas))
        self.patches.clear()
        
        return (result.cpu().numpy() * 255).astype(np.uint8)

def load_model():
    print(f"Loading : {MODEL_NAME} | device={DEVICE.upper()}")
    
    processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto" 
    )
    model.eval()
    
    print("Model ready.\n")
    return processor, model

def answer_question(processor, model, image: Image.Image, question: str, options: list) -> str:
    valid = [(str(i+1), opt) for i, opt in enumerate(options) if pd.notna(opt) and str(opt).strip() != ""]
    options_text = "\n".join([f"{num}) {opt}" for num, opt in valid])
    
    prompt_text = (
    f"You are a precise visual reasoning AI analyzing an OpenStreetMap image.\n\n"
    f"ORIENTATION:\n"
    f"- Top of the image = North\n"
    f"- Bottom = South\n"
    f"- Right = East\n"
    f"- Left = West\n\n"
    f"QUESTION:\n{question}\n\n"
    f"OPTIONS:\n{options_text}\n\n"
    f"INSTRUCTIONS:\n"
    f"1. Locate relevant landmarks visually.\n"
    f"2. Determine their relative positions using the defined orientation.\n"
    f"3. Base your answer only on clearly visible evidence.\n"
    f"4. Keep reasoning brief.\n"
    f"5. Eliminate incorrect options.\n"
    f"6. If uncertain, choose 5.\n\n"
    f"OUTPUT FORMAT:\n"
    f"Final Answer: [Number]"
    )
    
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

    output = model.generate(
        **inputs, 
        max_new_tokens=250, 
        do_sample=False,   
        pad_token_id=processor.tokenizer.eos_token_id
    )
    
    generated_text = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    match = re.search(r'Final Answer:\s*([1-5])', generated_text, re.IGNORECASE)
    
    if match:
        pred_num = match.group(1)
    else:
        fallback_match = re.findall(r'\b([1-5])\b', generated_text)
        if fallback_match:
            pred_num = fallback_match[-1]
        else:
            pred_num = "5"
            
    print(f"\n[Reasoning]: {generated_text.replace(chr(10), ' ')}") 
    return pred_num

def run_pipeline(image_path: str, csv_path: str, output_path: str):
    print(f"Image : {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    df = pd.read_csv(csv_path)
    processor, model = load_model()
    
    results = []
    for idx, row in df.iterrows():
        options = [row.get(f"option_{i}") for i in range(1, 5)]
        options.append("Information not present on the map.")
        
        num_ans = answer_question(processor, model, image, row["question"], options)
        print(f"[{idx+1:02d}] pred={num_ans}")
        
        question_id = row["question_id"]
        results.append({
            "id": question_id, 
            "question_num": question_id,
            "option": num_ans
        })
        
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Predictions saved successfully to {output_path}")

if __name__ == "__main__":
    # Use a single argument for the parent directory containing the data
    parser = argparse.ArgumentParser(description="Map stitching and VQA pipeline.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the directory containing test.csv and patches/")
    args = parser.parse_args()

    csv_path = os.path.join(args.test_dir, "test.csv")
    patch_dir = os.path.join(args.test_dir, "patches")

    print(f"--- Starting Map Assembly from {patch_dir} ---")
    
    if not os.path.exists(patch_dir):
        print(f"Error: Patches directory not found at {patch_dir}")
        exit(1)
        
    if not os.path.exists(csv_path):
        print(f"Error: test.csv not found at {csv_path}")
        exit(1)

    stitcher = ProductionMapStitcher(patch_dir, overlap_ratio=0.10, search_ratio=1.0)
    stitcher.run()
    map_final = stitcher.assemble()
    
    if map_final is not None:
        cv2.imwrite(MAP_IMAGE_PATH, map_final)
        print(f"Success: Production-grade map saved to {MAP_IMAGE_PATH}.\n")
        
        print("--- Starting LLaVA VQA Pipeline ---")
        run_pipeline(MAP_IMAGE_PATH, csv_path, OUTPUT_PATH)
    else:
        print("Error: Stitching returned None. VQA Pipeline aborted.")