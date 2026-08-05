#!/usr/bin/env python3
"""
GroundingDINO to YOLO Format Converter

This script converts GroundingDINO detection results to YOLO annotation format.
It can process single images or entire directories.

Features:
- Converts bounding boxes to YOLO format (normalized x_center, y_center, width, height)
- Keeps only the highest confidence detection per image
- Supports single image or batch directory processing
- Generates class mapping file
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Add GroundingDINO imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def calculate_iou(box1, box2):
    """
    計算兩個bounding box的IoU (Intersection over Union)
    
    Args:
        box1, box2: [x_center, y_center, width, height] 格式的box (歸一化座標 0-1)
    
    Returns:
        float: IoU值 (0-1)
    """
    # 轉換為 [x1, y1, x2, y2] 格式
    def center_to_corners(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    box1_corners = center_to_corners(box1)
    box2_corners = center_to_corners(box2)
    
    x1_1, y1_1, x2_1, y2_1 = box1_corners
    x1_2, y1_2, x2_2, y2_2 = box2_corners
    
    # 計算交集區域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 如果沒有交集，返回0
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # 計算交集面積
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 計算各自面積
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 計算聯集面積
    union = area1 + area2 - intersection
    
    # 計算IoU
    iou = intersection / union if union > 0 else 0.0
    return iou


def calculate_box_area(box):
    """
    計算bounding box的面積
    
    Args:
        box: [x_center, y_center, width, height] 格式的box (歸一化座標 0-1)
    
    Returns:
        float: 面積
    """
    _, _, width, height = box
    return width * height


def remove_duplicate_detections(boxes, pred_phrases, confidences, iou_threshold=0.5):
    """
    使用IoU去除重複檢測，保留面積最大的檢測框
    
    Args:
        boxes: torch.Tensor - 檢測框 [N, 4]
        pred_phrases: list - 預測標籤
        confidences: torch.Tensor - 信心度
        iou_threshold: float - IoU閾值，超過此值被認為是重複檢測
    
    Returns:
        tuple: 過濾後的 (boxes, pred_phrases, confidences)
    """
    if len(boxes) <= 1:
        return boxes, pred_phrases, confidences
    
    # 轉換為numpy以便操作
    boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else np.array(boxes)
    confidences_np = confidences.numpy() if isinstance(confidences, torch.Tensor) else np.array(confidences)
    
    # 計算所有box的面積
    areas = [calculate_box_area(box) for box in boxes_np]
    
    # 保留的檢測索引
    keep_indices = []
    used = [False] * len(boxes_np)
    
    # 按面積從大到小排序
    sorted_indices = sorted(range(len(boxes_np)), key=lambda i: areas[i], reverse=True)
    
    for i in sorted_indices:
        if used[i]:
            continue
            
        # 保留當前檢測
        keep_indices.append(i)
        used[i] = True
        
        # 檢查與其他檢測的IoU
        for j in range(len(boxes_np)):
            if i != j and not used[j]:
                iou = calculate_iou(boxes_np[i], boxes_np[j])
                if iou > iou_threshold:
                    # 標記為已使用（被抑制）
                    used[j] = True
                    print(f"  Suppressed detection {j+1} (IoU: {iou:.3f} with detection {i+1})")
    
    # 返回保留的檢測
    if keep_indices:
        filtered_boxes = boxes_np[keep_indices]
        filtered_phrases = [pred_phrases[i] for i in keep_indices]
        filtered_confidences = confidences_np[keep_indices]
        
        # 轉換回torch tensor
        filtered_boxes = torch.tensor(filtered_boxes)
        filtered_confidences = torch.tensor(filtered_confidences)
        
        print(f"  Kept {len(keep_indices)} detections after IoU filtering (removed {len(boxes) - len(keep_indices)} duplicates)")
        
        return filtered_boxes, filtered_phrases, filtered_confidences
    else:
        return boxes, pred_phrases, confidences


def load_image(image_path):
    """Load and preprocess image for GroundingDINO."""
    image_pil = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.RandomResize([512], max_size=768),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    """Load GroundingDINO model."""
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loaded: {load_res}")
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, 
                        cpu_only=False, token_spans=None):
    """Get detection results from GroundingDINO."""
    assert text_threshold is not None or token_spans is not None, \
        "text_threshold and token_spans should not be None at the same time!"
    
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    
    # Filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        # Get confidence scores
        confidences = logits_filt.max(dim=1)[0]
        
        # Get phrase labels
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        
        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
    else:
        # Token spans mode (not implemented in this version)
        raise NotImplementedError("Token spans mode not implemented")
    
    return boxes_filt, pred_phrases, confidences


def convert_to_yolo_format(boxes, image_width, image_height):
    """Convert GroundingDINO boxes to YOLO format."""
    yolo_boxes = []
    
    for box in boxes:
        # GroundingDINO outputs normalized coordinates (0-1)
        # Convert from center format to corner format, then to YOLO format
        x_center, y_center, width, height = box
        
        # Ensure coordinates are within valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_boxes.append([x_center, y_center, width, height])
    
    return yolo_boxes


def get_class_mapping(classes_file=None):
    """Get or create class mapping."""
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return {cls: idx for idx, cls in enumerate(classes)}
    else:
        # Default mapping for common screen detection classes
        default_classes = ['screen', 'monitor', 'display', 'born_screen1', 'born_screen2', 'born_screen3']
        return {cls: idx for idx, cls in enumerate(default_classes)}


def process_single_image(image_path, model, text_prompt, box_threshold, text_threshold,
                        output_dir, class_mapping, cpu_only=False, keep_all=True, iou_threshold=0.5):
    """Process a single image and generate YOLO annotation."""
    try:
        # Load image
        image_pil, image = load_image(image_path)
        image_width, image_height = image_pil.size
        
        # Get detections
        boxes, pred_phrases, confidences = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only
        )
        
        if len(boxes) == 0:
            print(f"No detections found in {image_path}")
            return False
        
        # Apply IoU filtering to remove duplicate detections
        print(f"Original detections: {len(boxes)}")
        boxes, pred_phrases, confidences = remove_duplicate_detections(
            boxes, pred_phrases, confidences, iou_threshold=iou_threshold
        )
        print(f"After IoU filtering: {len(boxes)} detections")
        
        if keep_all:
            # Keep all detections (after IoU filtering)
            print(f"Found {len(boxes)} unique detections for {os.path.basename(image_path)}:")
            
            # Convert to YOLO format
            yolo_boxes = convert_to_yolo_format(boxes, image_width, image_height)
            
            # Generate output filename
            image_name = Path(image_path).stem
            output_file = os.path.join(output_dir, f"{image_name}.txt")
            
            # Write YOLO annotation for all detections
            with open(output_file, 'w') as f:
                for i, (yolo_box, phrase, confidence) in enumerate(zip(yolo_boxes, pred_phrases, confidences)):
                    # Determine class ID for each detection
                    class_id = 0  # Default class
                    for class_name, class_idx in class_mapping.items():
                        if class_name.lower() in phrase.lower():
                            class_id = class_idx
                            break
                    
                    x_center, y_center, width, height = yolo_box
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    print(f"  Detection {i+1}: {phrase} (confidence: {confidence:.3f}) -> class {class_id}")
        else:
            # Keep only the highest confidence detection (after IoU filtering)
            best_idx = torch.argmax(confidences)
            best_box = boxes[best_idx:best_idx+1]
            best_phrase = pred_phrases[best_idx]
            best_confidence = confidences[best_idx]
            
            print(f"Best detection for {os.path.basename(image_path)}: {best_phrase} (confidence: {best_confidence:.3f})")
            
            # Convert to YOLO format
            yolo_boxes = convert_to_yolo_format(best_box, image_width, image_height)
            
            # Determine class ID
            class_id = 0  # Default class
            for class_name, class_idx in class_mapping.items():
                if class_name.lower() in best_phrase.lower():
                    class_id = class_idx
                    break
            
            # Generate output filename
            image_name = Path(image_path).stem
            output_file = os.path.join(output_dir, f"{image_name}.txt")
            
            # Write YOLO annotation
            with open(output_file, 'w') as f:
                for yolo_box in yolo_boxes:
                    x_center, y_center, width, height = yolo_box
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Annotation saved: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


def process_directory(input_dir, model, text_prompt, box_threshold, text_threshold,
                     output_dir, class_mapping, cpu_only=False, keep_all=True, iou_threshold=0.5):
    """Process all images in a directory."""
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images in directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    success_count = 0
    for image_file in image_files:
        if process_single_image(str(image_file), model, text_prompt, box_threshold, 
                              text_threshold, output_dir, class_mapping, cpu_only, keep_all, iou_threshold):
            success_count += 1
    
    print(f"\nProcessing complete: {success_count}/{len(image_files)} images processed successfully")


def save_class_mapping(class_mapping, output_dir):
    """Save class mapping to classes.txt file."""
    classes_file = os.path.join(output_dir, "classes.txt")
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    
    with open(classes_file, 'w') as f:
        for class_name, _ in sorted_classes:
            f.write(f"{class_name}\n")
    
    print(f"Class mapping saved: {classes_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert GroundingDINO detections to YOLO format")
    
    # Model arguments
    parser.add_argument("--config_file", "-c", type=str, required=True,
                       help="Path to GroundingDINO config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True,
                       help="Path to GroundingDINO checkpoint file")
    
    # Input arguments
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input image file or directory")
    parser.add_argument("--text_prompt", "-t", type=str, required=True,
                       help="Text prompt for detection")
    
    # Output arguments
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="Output directory for YOLO annotations")
    
    # Detection parameters
    parser.add_argument("--box_threshold", type=float, default=0.3,
                       help="Box threshold for detection")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                       help="Text threshold for detection")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                       help="IoU threshold for removing duplicate detections (0.0-1.0)")
    
    # Optional arguments
    parser.add_argument("--classes_file", type=str, default=None,
                       help="Path to classes.txt file")
    parser.add_argument("--cpu_only", action="store_true",
                       help="Run on CPU only")
    parser.add_argument("--keep_all", action="store_true",
                       help="Keep all detections instead of only the highest confidence one")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading GroundingDINO model...")
    model = load_model(args.config_file, args.checkpoint_path, args.cpu_only)
    
    # Get class mapping
    class_mapping = get_class_mapping(args.classes_file)
    print(f"Class mapping: {class_mapping}")
    
    # Save class mapping
    save_class_mapping(class_mapping, args.output_dir)
    
    # Process input
    if os.path.isfile(args.input):
        print(f"Processing single image: {args.input}")
        process_single_image(args.input, model, args.text_prompt, args.box_threshold,
                           args.text_threshold, args.output_dir, class_mapping, 
                           args.cpu_only, args.keep_all, args.iou_threshold)
    elif os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        process_directory(args.input, model, args.text_prompt, args.box_threshold,
                        args.text_threshold, args.output_dir, class_mapping, 
                        args.cpu_only, args.keep_all, args.iou_threshold)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
