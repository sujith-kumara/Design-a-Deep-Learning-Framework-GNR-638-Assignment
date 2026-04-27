
import os

class VQAEngine:
    def __init__(self, model_path="OpenGVLab/InternVL2-8B", device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Load in 8-bit or 4-bit to fit in memory
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            trust_remote_code=True,
            device_map='auto'
        ).eval()

    def answer_question(self, image_path, question, options):
        from PIL import Image
        pixel_values = self._load_image(image_path)
        prompt = self._build_prompt(question, options)
        
        # InternVL2 generation logic
        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config={"max_new_tokens": 10},
            history=None,
            return_history=True
        )
        return self._post_process(response)

    def _load_image(self, image_path, max_num=12):
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        # InternVL2 Dynamic Preprocessing to handle high-resolution without losing detail
        return self.model.preprocess_image(image, max_num=max_num)

    def _build_prompt(self, question, options):
        opts_str = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
        return (
            f"Image Analysis Task: Geospatial Navigation and Landmark Identification.\n"
            f"The image provided is a 1472x1472 pixel reconstructed map. Please zoom into the specific labels to find the answer.\n"
            f"Step 1: Scan the map for keywords from the question.\n"
            f"Step 2: Identify the relative position of the landmarks.\n"
            f"Step 3: Choose the correct option.\n\n"
            f"Question: {question}\n"
            f"Options:\n{opts_str}\n"
            f"Return your internal reasoning briefly, and end with the final choice: 'Final Answer: [Number]'. "
            f"If uncertain, output 'Final Answer: 5'."
        )

    def _post_process(self, response):
        import re
        # Look for the 'Final Answer' pattern specifically
        match = re.search(r'Final Answer:\s*(\d)', response)
        if match:
            return int(match.group(1))
        # Fallback to any digit
        digits = re.findall(r'\d', response)
        if digits:
            val = int(digits[-1]) # Often the last digit is the choice
            return val if val in [1, 2, 3, 4, 5] else 5
        return 5

class MockVQAEngine:
    def answer_question(self, image_path, question, options):
        print(f"Mock answering: {question}")
        return 1

if __name__ == "__main__":
    engine = MockVQAEngine()
    ans = engine.answer_question("reconstructed_map.png", "Where is Vihar Lake?", ["North", "South", "East", "West"])
    print(f"Answer: {ans}")
