"""
Inference utilities for the EfficientNet-B0 classifier trained
in `efficientnet_b0_image_classification.ipynb`.

The training notebook saves a checkpoint dict that includes:
    model_state_dict, class_names, num_classes, img_size,
    normalize_mean, normalize_std, best_val_acc, history, arch

This module loads that checkpoint and exposes a single `predict()`
method that turns a PIL image into a ranked list of class probabilities.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image


class ImageClassifier:
    """Wraps the trained EfficientNet-B0 model for inference."""

    def __init__(self, checkpoint_path: str | Path, device: Optional[torch.device] = None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Set the MODEL_PATH environment variable or place the .pth file there."
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # `weights_only=False` is needed because the checkpoint contains the
        # `history` dict and class-name list alongside the state dict.
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        self.class_names: List[str] = list(ckpt["class_names"])
        self.num_classes: int = int(ckpt["num_classes"])
        self.img_size: int = int(ckpt.get("img_size", 224))
        self.best_val_acc: float = float(ckpt.get("best_val_acc", 0.0))

        mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
        std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])

        # Rebuild the architecture and load the trained weights
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def predict(self, image: Image.Image, top_k: Optional[int] = None) -> List[Dict]:
        """Return a list of `{class, probability}` dicts, sorted descending."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_k = top_k or len(self.class_names)
        idx_sorted = probs.argsort()[::-1][:top_k]
        return [
            {"class": self.class_names[i], "probability": float(probs[i])}
            for i in idx_sorted
        ]

    def predict_from_bytes(self, raw: bytes, top_k: Optional[int] = None) -> List[Dict]:
        """Convenience wrapper for raw upload bytes."""
        image = Image.open(io.BytesIO(raw))
        return self.predict(image, top_k=top_k)
