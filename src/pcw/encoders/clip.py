"""Optional frozen local CLIP encoder; construction explicitly fetches weights."""
from typing import Protocol
import numpy as np
import torch

class Encoder(Protocol):
    def encode(self, images:list[np.ndarray])->np.ndarray: ...

class ClipEncoder:
    def __init__(self, model="openai/clip-vit-base-patch32",device="auto"):
        from transformers import CLIPModel, CLIPProcessor
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else torch.device(device)
        self.processor=CLIPProcessor.from_pretrained(model)
        self.model=CLIPModel.from_pretrained(model).to(self.device).eval()
        self.model.requires_grad_(False)
    def encode(self,images):
        batch=self.processor(images=images,return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,enabled=self.device.type=="cuda"):
            x=self.model.get_image_features(**batch)
            x=torch.nn.functional.normalize(x.float(),dim=-1)
        return x.cpu().numpy()
