
import pandas as pd
import os
from stitcher import MapStitcher
from vqa_engine import VQAEngine, MockVQAEngine

def main():
    # 1. Stitch Map
    print("Step 1: Stitching patches...")
    patch_dir = "patches"
    if not os.path.exists(patch_dir):
        # On Kaggle, adapt path if needed
        patch_dir = "/kaggle/input/competition-name/patches"
    
    stitcher = MapStitcher(patch_dir)
    stitcher.stitch()
    reconstructed = stitcher.assemble()
    
    map_path = "reconstructed_map.png"
    import cv2
    cv2.imwrite(map_path, reconstructed)
    print(f"Map reconstructed and saved to {map_path}")

    # 2. Initialize VQA
    print("Step 2: Initializing VQA engine...")
    try:
        # Try real model if on Kaggle with GPU
        model_path = "/kaggle/input/internvl2-8b-python-weights" # Path to local weights in Kaggle Dataset
        engine = VQAEngine(model_path=model_path)
    except Exception as e:
        print(f"Could not load real model: {e}. Using mock.")
        engine = MockVQAEngine()

    # 3. Process Questions
    print("Step 3: Answering questions...")
    test_df = pd.read_csv("test.csv")
    results = []
    
    for i, row in test_df.iterrows():
        question = row['question']
        options = [row['option_1'], row['option_2'], row['option_3'], row['option_4']]
        
        answer = engine.answer_question(map_path, question, options)
        results.append({
            'id': row['id'],
            'question_num': row['id'],
            'option': answer
        })
    
    # 4. Save Submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
