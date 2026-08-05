import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_overlapping_boxes(boxes, labels, iou_threshold=0.3):
    """Merge overlapping bounding boxes."""
    if len(boxes) <= 1:
        return boxes, labels
    
    # Convert to numpy for easier manipulation
    boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else np.array(boxes)
    merged_boxes = []
    merged_labels = []
    used = [False] * len(boxes_np)
    
    for i in range(len(boxes_np)):
        if used[i]:
            continue
            
        current_box = boxes_np[i].copy()
        current_label = labels[i]
        used[i] = True
        
        # Find all boxes that overlap with current box
        for j in range(i + 1, len(boxes_np)):
            if used[j]:
                continue
                
            iou = calculate_iou(current_box, boxes_np[j])
            if iou > iou_threshold:
                # Merge boxes by taking the union
                current_box[0] = min(current_box[0], boxes_np[j][0])  # x_min
                current_box[1] = min(current_box[1], boxes_np[j][1])  # y_min
                current_box[2] = max(current_box[2], boxes_np[j][2])  # x_max
                current_box[3] = max(current_box[3], boxes_np[j][3])  # y_max
                used[j] = True
        
        merged_boxes.append(current_box)
        merged_labels.append(current_label)
    
    return torch.tensor(np.array(merged_boxes)), merged_labels


def expand_box_to_bezel(box, image_size, expansion_factor=0.05):
    """Expand bounding box to include screen bezel."""
    W, H = image_size
    x0, y0, x1, y1 = box
    
    # Calculate expansion
    width = x1 - x0
    height = y1 - y0
    expand_x = width * expansion_factor
    expand_y = height * expansion_factor
    
    # Expand box
    x0 = max(0, x0 - expand_x)
    y0 = max(0, y0 - expand_y)
    x1 = min(W, x1 + expand_x)
    y1 = min(H, y1 + expand_y)
    
    return [x0, y0, x1, y1]


def crop_and_save_screens(image_pil, boxes, labels, output_dir,prefix=""):
    """Crop detected screens and save them as separate images."""
    W, H = image_pil.size
    cropped_images = []
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Convert normalized coordinates to pixel coordinates
        box_pixels = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box_pixels[:2] -= box_pixels[2:] / 2
        box_pixels[2:] += box_pixels[:2]
        
        x0, y0, x1, y1 = box_pixels.int().tolist()
        
        # Expand box to include bezel
        expanded_box = expand_box_to_bezel([x0, y0, x1, y1], (W, H))
        x0, y0, x1, y1 = [int(coord) for coord in expanded_box]
        
        # Crop the image
        cropped = image_pil.crop((x0, y0, x1, y1))
        
        # Save cropped image
        # Use prefix directly as it should already be just the filename
        crop_filename = f"{prefix}-screen_crop_{i+1}.jpg" if prefix else f"screen_crop_{i+1}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        cropped.save(crop_path, quality=95, optimize=False)  # Fixed: Use high quality (95) instead of default 75
        cropped_images.append((cropped, crop_filename))
        
        print(f"Saved cropped screen {i+1} to {crop_filename}")
    
    return cropped_images


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Expand box to include bezel for visualization
        expanded_box = expand_box_to_bezel([x0, y0, x1, y1], (W, H))
        x0, y0, x1, y1 = [int(coord) for coord in expanded_box]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    
    # 將模型移到指定設備
    device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"模型已載入到設備: {device}")
    
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    device="cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO Screen Cropping", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for merging overlapping boxes")
    parser.add_argument("--bezel_expansion", type=float, default=0.05, help="Expansion factor for including screen bezel")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    iou_threshold = args.iou_threshold
    bezel_expansion = args.bezel_expansion

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
    )

    # Merge overlapping boxes
    if len(boxes_filt) > 1:
        print(f"Found {len(boxes_filt)} detections. Merging overlapping boxes...")
        merged_boxes, merged_labels = merge_overlapping_boxes(boxes_filt, pred_phrases, iou_threshold)
        print(f"After merging: {len(merged_boxes)} unique screens detected.")
    else:
        merged_boxes, merged_labels = boxes_filt, pred_phrases

    # Crop and save individual screens
    cropped_images = crop_and_save_screens(image_pil, merged_boxes, merged_labels, output_dir)

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": merged_boxes,
        "size": [size[1], size[0]],  # H,W
        "labels": merged_labels,
    }
    
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, "pred_merged.jpg"))
    
    print(f"\nResults saved to {output_dir}:")
    print(f"- Original image: raw_image.jpg")
    print(f"- Prediction with merged boxes: pred_merged.jpg")
    for i, (_, filename) in enumerate(cropped_images):
        print(f"- Cropped screen {i+1}: {filename}")
