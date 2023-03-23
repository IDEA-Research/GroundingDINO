from typing import Tuple, List

import numpy as np
import torch
from PIL.Image import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.cuda()
    image = image.cuda()

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    pred_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    mask = pred_logits.max(dim=1)[0] > box_threshold
    logits = pred_logits[mask]  # num_filt, 256
    boxes = pred_boxes[mask]  # num_filt, 4

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, caption).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases
