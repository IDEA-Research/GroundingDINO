#!/usr/bin/env python3
"""
Simple Batch Screen Cropper

A streamlined script specifically for batch processing folders of medical images.
This script focuses on speed and simplicity for processing large numbers of images.

Usage:
    python batch_screen_crop.py input_folder output_folder

Features:
- Processes all images in a folder
- Efficient model loading (once for all images)
- Progress tracking
- Error handling
- Automatic folder organization
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_screen_crop import *


def get_image_files(folder_path):
    """Get all image files from a folder."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    extensions += [ext.upper() for ext in extensions]
    
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Batch process medical images for screen detection and cropping")
    parser.add_argument("input_folder", help="Input folder containing images")
    parser.add_argument("output_folder", help="Output folder for results")
    parser.add_argument("--text_prompt", default="screen", help="Detection prompt (default: 'screen')")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Detection threshold (default: 0.3)")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for merging (default: 0.3)")
    parser.add_argument("--bezel_expansion", type=float, default=0.05, help="Bezel expansion factor (default: 0.05)")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only")
    
    args = parser.parse_args()
    
    # Check input folder
    if not os.path.exists(args.input_folder):
        print(f"❌ Error: Input folder not found: {args.input_folder}")
        return 1
    
    # Get image files
    image_files = get_image_files(args.input_folder)
    if not image_files:
        print(f"❌ No image files found in {args.input_folder}")
        return 1
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Configuration paths
    config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "/home/aic/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth"
    
    if not os.path.exists(config_file) or not os.path.exists(checkpoint_path):
        print("❌ Model files not found. Please check config and checkpoint paths.")
        return 1
    
    print(f"🚀 Batch Screen Cropping")
    print(f"📁 Input: {args.input_folder}")
    print(f"📂 Output: {args.output_folder}")
    print(f"📸 Images: {len(image_files)}")
    print(f"🎯 Prompt: '{args.text_prompt}'")
    print(f"⚙️  Thresholds: box={args.box_threshold}, iou={args.iou_threshold}, bezel={args.bezel_expansion}")
    print("=" * 60)
    
    # Load model once
    print("🤖 Loading model...")
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    
    # Process images
    successful = 0
    total_screens = 0
    
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        print(f"[{i:3d}/{len(image_files)}] {image_name}...", end=" ")
        
        try:
            # Create output subfolder
            output_subfolder = os.path.join(args.output_folder, image_name)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # Load and process image
            image_pil, image = load_image(image_path)
            
            # Run detection
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, args.text_prompt, args.box_threshold, 0.25,
                cpu_only=args.cpu_only, token_spans=None
            )
            
            if len(boxes_filt) == 0:
                print("⚠️  No screens")
                continue
            
            # Merge overlapping boxes
            if len(boxes_filt) > 1:
                merged_boxes, merged_labels = merge_overlapping_boxes(
                    boxes_filt, pred_phrases, args.iou_threshold
                )
            else:
                merged_boxes, merged_labels = boxes_filt, pred_phrases
            
            # Save original
            image_pil.save(os.path.join(output_subfolder, "original.jpg"))
            
            # Crop screens
            cropped_images = crop_and_save_screens(
                image_pil, merged_boxes, merged_labels, output_subfolder
            )
            
            # Save visualization
            size = image_pil.size
            pred_dict = {
                "boxes": merged_boxes,
                "size": [size[1], size[0]],
                "labels": merged_labels,
            }
            vis_image = plot_boxes_to_image(image_pil, pred_dict)[0]
            vis_image.save(os.path.join(output_subfolder, "detection.jpg"))
            
            screens_count = len(cropped_images)
            total_screens += screens_count
            successful += 1
            print(f"✅ {screens_count} screens")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("=" * 60)
    print(f"📊 SUMMARY")
    print(f"✅ Successful: {successful}/{len(image_files)}")
    print(f"📱 Total screens: {total_screens}")
    print(f"📂 Results: {args.output_folder}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
