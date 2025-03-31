from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from config import BLIP_MODEL

class BLIPCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(self.device)

    def generate_caption(self, image_path):
        try:
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            return self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise Exception(f"Caption generation failed: {str(e)}")