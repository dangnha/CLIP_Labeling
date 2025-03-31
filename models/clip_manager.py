import clip
import torch
from PIL import Image
import numpy as np
from config import CLIP_MODEL

class CLIPManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)
        
    def classify_image(self, image_path, categories):
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text = clip.tokenize(categories).to(self.device)
            
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                
            return {cat: float(prob) for cat, prob in zip(categories, probs)}
        except Exception as e:
            raise Exception(f"Classification failed: {str(e)}")
    
    def search_by_image(self, query_path, images):
        try:
            query_image = self.preprocess(Image.open(query_path)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                query_features = self.model.encode_image(query_image)
                query_features /= query_features.norm(dim=-1, keepdim=True)
                
                results = []
                for img in images:
                    image = self.preprocess(Image.open(img['path'])).unsqueeze(0).to(self.device)
                    features = self.model.encode_image(image)
                    features /= features.norm(dim=-1, keepdim=True)
                    similarity = (query_features @ features.T).item()
                    results.append({
                        **img,
                        'similarity': similarity
                    })
                    
            return sorted(results, key=lambda x: x['similarity'], reverse=True)[:10]
        except Exception as e:
            raise Exception(f"Image search failed: {str(e)}")
    
    def search_by_text(self, query, images):
        try:
            text = clip.tokenize([query]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                results = []
                for img in images:
                    image = self.preprocess(Image.open(img['path'])).unsqueeze(0).to(self.device)
                    features = self.model.encode_image(image)
                    features /= features.norm(dim=-1, keepdim=True)
                    similarity = (text_features @ features.T).item()
                    results.append({
                        **img,
                        'similarity': similarity
                    })
                    
            return sorted(results, key=lambda x: x['similarity'], reverse=True)[:10]
        except Exception as e:
            raise Exception(f"Text search failed: {str(e)}")