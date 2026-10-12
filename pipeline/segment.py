"""Semantic segmentation (SegFormer ADE20k) collapsed to sky / ground / building / ignore."""
import os

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"
SKY, GROUND, BUILDING, IGNORE = 0, 1, 2, 255

# ADE20k ids
_BUILDING = [0, 1, 25, 48, 79, 84, 86]  # wall, building, house, skyscraper, hovel, tower, awning
_GROUND = [3, 6, 11, 13, 29, 46, 52, 91, 94]  # floor, road, sidewalk, earth, field, sand, path, dirt track, land
_SKY = [2]

_LUT = np.full(150, IGNORE, np.uint8)
_LUT[_BUILDING] = BUILDING
_LUT[_GROUND] = GROUND
_LUT[_SKY] = SKY


class Segmenter:
    def __init__(self, threads=None):
        if threads:
            torch.set_num_threads(threads)
        self.proc = SegformerImageProcessor.from_pretrained(MODEL)
        self.device = torch.device(os.environ.get("SEG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval().to(self.device)

    @torch.no_grad()
    def probs(self, img: Image.Image, size=512):
        """Per-pixel class probabilities (150, H, W) at the image's resolution (downscaled to <= size)."""
        w, h = img.size
        s = size / max(w, h)
        img = img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        x = self.proc(images=img, return_tensors="pt", do_resize=False).to(self.device)
        logits = self.model(**x).logits
        logits = torch.nn.functional.interpolate(logits, size=(img.height, img.width), mode="bilinear", align_corners=False)
        return logits.softmax(1)[0].cpu().numpy()

    def labels(self, img, size=512):
        return collapse(self.probs(img, size).argmax(0))


def collapse(ade):
    return _LUT[ade]
