import torch
import clip
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def encode_image(self, images) -> np.ndarray:
        pass


class CLIPModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def encode_text(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)
            return text_features

    def encode_image(self, images) -> np.ndarray:
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                if isinstance(images, list):
                    image_tensors = torch.stack(
                        [self.preprocess(img) for img in images]
                    ).to(self.device)
                else:
                    image_tensors = self.preprocess(
                        images).unsqueeze(0).to(self.device)
            else:
                image_tensors = images.to(self.device)

            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()


if __name__ == "__main__":
    pass
