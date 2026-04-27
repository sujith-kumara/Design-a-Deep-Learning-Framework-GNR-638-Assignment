
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

    def _load_image(self, image_path):
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        return self.model.preprocess_image(image)

    def _build_prompt(self, question, options):
        opts_str = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
        return (
            f"You are looking at a high-resolution geospatial map of Mumbai. \n"
            f"Please carefully examine the labels, landmarks, and spatial relationships in the image to answer the following question.\n"
            f"Question: {question}\n"
            f"Options:\n{opts_str}\n"
            f"Output only the option number (1, 2, 3, 4, or 5). Select 5 only if you are completely unable to find the answer."
        )

    def _post_process(self, response):
        import re
        digits = re.findall(r'\d', response)
        if digits:
            val = int(digits[0])
            if val in [1, 2, 3, 4, 5]:
                return val
        return 5

class MockVQAEngine:
    def answer_question(self, image_path, question, options):
        print(f"Mock answering: {question}")
        return 1

if __name__ == "__main__":
    engine = MockVQAEngine()
    ans = engine.answer_question("reconstructed_map.png", "Where is Vihar Lake?", ["North", "South", "East", "West"])
    print(f"Answer: {ans}")
