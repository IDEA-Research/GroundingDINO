#!/usr/bin/env python3
"""
Screen Detection and Cropping Demo

This script demonstrates how to detect screens in medical images and automatically crop them.
It merges overlapping detections and expands bounding boxes to include screen bezels.

Usage:
    Single image: python screen_crop_demo.py --image_path your_image.jpg
    Batch folder: python screen_crop_demo.py --folder_path your_folder/

The script will:
1. Detect all screens in the image(s)
2. Merge overlapping detections into single bounding boxes
3. Expand boxes to include screen bezels
4. Save cropped individual screen images
5. Save a visualization with merged bounding boxes
"""

import argparse
import os
import sys
import glob
from pathlib import Path

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_screen_crop import *


def get_image_files(folder_path):
    """Get all image files from a folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(folder_path, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    return sorted(image_files)


def process_single_image(image_path, output_dir, config_file, checkpoint_path, args, model=None):
    """Process a single image and return processing results."""
    print(f"\n📸 Processing: {os.path.basename(image_path)}")
    
    # Create subdirectory for this image
    image_name = Path(image_path).stem
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    try:
        # Load image
        image_pil, image = load_image(image_path)
        
        # Load model if not provided
        if model is None:
            model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
        
        # Save original image
        image_pil.save(os.path.join(image_output_dir, "original_image.jpg"))
        
        # Run detection
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, args.text_prompt, args.box_threshold, args.text_threshold, 
            cpu_only=args.cpu_only, token_spans=None
        )
        
        if len(boxes_filt) == 0:
            print(f"  ❌ No screens detected in {os.path.basename(image_path)}")
            return {
                'status': 'no_detection',
                'screens_detected': 0,
                'image_path': image_path,
                'output_dir': image_output_dir
            }
        
        print(f"  🎯 Found {len(boxes_filt)} initial detections")
        
        # Merge overlapping boxes
        if len(boxes_filt) > 1:
            merged_boxes, merged_labels = merge_overlapping_boxes(boxes_filt, pred_phrases, args.iou_threshold)
            print(f"  🔄 After merging: {len(merged_boxes)} unique screens")
        else:
            merged_boxes, merged_labels = boxes_filt, pred_phrases
        
        # Crop and save individual screens
        cropped_images = crop_and_save_screens(image_pil, merged_boxes, merged_labels, image_output_dir)
        
        # Create visualization with merged boxes
        size = image_pil.size
        pred_dict = {
            "boxes": merged_boxes,
            "size": [size[1], size[0]],  # H,W
            "labels": merged_labels,
        }
        
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(image_output_dir, "detected_screens.jpg"))
        
        print(f"  ✅ Saved {len(cropped_images)} cropped screens")
        
        return {
            'status': 'success',
            'screens_detected': len(cropped_images),
            'image_path': image_path,
            'output_dir': image_output_dir,
            'cropped_files': [filename for _, filename in cropped_images]
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {os.path.basename(image_path)}: {str(e)}")
        return {
            'status': 'error',
            'screens_detected': 0,
            'image_path': image_path,
            'output_dir': image_output_dir,
            'error': str(e)
        }


def process_folder(folder_path, output_dir, config_file, checkpoint_path, args):
    """Process all images in a folder."""
    print(f"🔍 Scanning folder: {folder_path}")
    
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print("❌ No image files found in the specified folder.")
        return
    
    print(f"📁 Found {len(image_files)} image files")
    print(f"📂 Output directory: {output_dir}")
    print("-" * 60)
    
    # Load model once for all images
    print("🤖 Loading model...")
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    
    # Process each image
    results = []
    total_screens = 0
    successful_images = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        result = process_single_image(image_path, output_dir, config_file, checkpoint_path, args, model)
        results.append(result)
        
        if result['status'] == 'success':
            successful_images += 1
            total_screens += result['screens_detected']
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"📁 Total images processed: {len(image_files)}")
    print(f"✅ Successfully processed: {successful_images}")
    print(f"❌ Failed: {len(image_files) - successful_images}")
    print(f"📱 Total screens detected: {total_screens}")
    print(f"📂 Results saved to: {output_dir}")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for result in results:
        status_icon = "✅" if result['status'] == 'success' else "❌" if result['status'] == 'error' else "⚠️"
        image_name = os.path.basename(result['image_path'])
        if result['status'] == 'success':
            print(f"  {status_icon} {image_name}: {result['screens_detected']} screens")
        elif result['status'] == 'no_detection':
            print(f"  {status_icon} {image_name}: No screens detected")
        else:
            print(f"  {status_icon} {image_name}: Error - {result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser("Medical Screen Detection and Cropping Demo", add_help=True)
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", "-i", type=str, help="path to single input image file")
    input_group.add_argument("--folder_path", "-f", type=str, help="path to folder containing images to process")
    
    parser.add_argument("--output_dir", "-o", type=str, default="screen_crops", help="output directory for cropped screens")
    parser.add_argument("--text_prompt", "-t", type=str, default="screen", help="text prompt for detection")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="detection confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text matching threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for merging overlapping boxes")
    parser.add_argument("--bezel_expansion", type=float, default=0.05, help="Expansion factor for including screen bezel (0.05 = 5%)")
    parser.add_argument("--cpu-only", action="store_true", help="run inference on CPU only")
    
    args = parser.parse_args()
    
    # Configuration paths (you may need to adjust these)
    config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "/home/aic/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth"
    
    # Check if config and checkpoint files exist
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print("Please download the model weights or update the checkpoint_path.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process single image or folder
    if args.image_path:
        # Single image processing
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            return
        
        print(f"🖼️ Single Image Mode")
        print(f"📸 Processing image: {args.image_path}")
        print(f"📂 Output directory: {args.output_dir}")
        print(f"🎯 Detection prompt: '{args.text_prompt}'")
        print(f"📊 Box threshold: {args.box_threshold}")
        print(f"🔗 IoU threshold: {args.iou_threshold}")
        print(f"📐 Bezel expansion: {args.bezel_expansion * 100:.1f}%")
        print("-" * 50)
        
        result = process_single_image(args.image_path, args.output_dir, config_file, checkpoint_path, args)
        
        if result['status'] == 'success':
            print(f"\n✅ Processing complete!")
            print(f"📱 Detected {result['screens_detected']} screens")
            print(f"📂 Results saved to: {result['output_dir']}")
        elif result['status'] == 'no_detection':
            print(f"\n⚠️ No screens detected. Try lowering the box_threshold.")
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    else:
        # Folder processing
        if not os.path.exists(args.folder_path):
            print(f"Error: Folder not found: {args.folder_path}")
            return
        
        print(f"📁 Batch Processing Mode")
        print(f"🎯 Detection prompt: '{args.text_prompt}'")
        print(f"📊 Box threshold: {args.box_threshold}")
        print(f"🔗 IoU threshold: {args.iou_threshold}")
        print(f"📐 Bezel expansion: {args.bezel_expansion * 100:.1f}%")
        
        process_folder(args.folder_path, args.output_dir, config_file, checkpoint_path, args)


if __name__ == "__main__":
    main()
