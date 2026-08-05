# Screen Detection and Cropping for Medical Images

This enhanced version of GroundingDINO's inference script provides automatic screen detection, merging of overlapping detections, and cropping of individual screens from medical images. Now supports both single image and batch folder processing.

## New Features

1. **Overlapping Box Merging**: Automatically merges multiple detections of the same screen into a single bounding box
2. **Bezel Expansion**: Expands bounding boxes to include screen bezels for complete screen capture
3. **Individual Screen Cropping**: Saves each detected screen as a separate image file
4. **Improved Visualization**: Shows merged bounding boxes instead of multiple overlapping boxes
5. **Batch Processing**: Process entire folders of images automatically
6. **Organized Output**: Creates separate subdirectories for each processed image

## Files

- `inference_screen_crop.py` - Enhanced inference script with merging and cropping capabilities
- `screen_crop_demo.py` - User-friendly demo script with single image and batch processing
- `SCREEN_CROPPING_README.md` - This documentation file

## Usage

### Single Image Processing

```bash
python screen_crop_demo.py -i your_image.jpg -o output_folder
```

### Batch Folder Processing

```bash
python screen_crop_demo.py -f /path/to/image/folder/ -o output_folder
```

**Common Options:**
- `-i, --image_path`: Path to single input image (for single image mode)
- `-f, --folder_path`: Path to folder containing images (for batch mode)
- `-o, --output_dir`: Output directory (default: "screen_crops")
- `-t, --text_prompt`: Detection prompt (default: "screen")
- `--box_threshold`: Detection confidence threshold (default: 0.3)
- `--iou_threshold`: IoU threshold for merging overlapping boxes (default: 0.3)
- `--bezel_expansion`: Expansion factor for screen bezel (default: 0.05 = 5%)
- `--cpu-only`: Run on CPU instead of GPU

### Advanced Usage Examples

**Single image with custom settings:**
```bash
python screen_crop_demo.py -i medical_frame.jpg -o results \
  --box_threshold 0.25 \
  --iou_threshold 0.2 \
  --bezel_expansion 0.08
```

**Batch processing with high sensitivity:**
```bash
python screen_crop_demo.py -f /path/to/medical/images/ -o batch_results \
  --box_threshold 0.2 \
  --text_prompt "monitor screen"
```

## Batch Processing Features

### Automatic File Discovery
- Scans folder for common image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- Case-insensitive extension matching
- Processes files in alphabetical order

### Organized Output Structure
```
output_folder/
├── image1_name/
│   ├── original_image.jpg
│   ├── detected_screens.jpg
│   ├── screen_crop_1.jpg
│   ├── screen_crop_2.jpg
│   └── ...
├── image2_name/
│   ├── original_image.jpg
│   ├── detected_screens.jpg
│   └── ...
└── ...
```

### Progress Tracking
- Real-time progress indicators
- Per-image status reporting
- Comprehensive summary statistics
- Error handling and reporting

### Batch Processing Summary
After processing, you'll get a detailed summary including:
- Total images processed
- Success/failure counts
- Total screens detected across all images
- Per-image detailed results

## Key Parameters

### IoU Threshold (`--iou_threshold`)
- Controls how much overlap is needed to merge two detections
- Range: 0.0 to 1.0
- Lower values = more aggressive merging
- Default: 0.3 (30% overlap)

### Bezel Expansion (`--bezel_expansion`) 
- Expands bounding boxes to include screen bezels
- Range: 0.0 to 1.0 (as fraction of screen size)
- 0.05 = expand by 5% of screen width/height in each direction
- Default: 0.05

### Box Threshold (`--box_threshold`)
- Minimum confidence for initial detection
- Range: 0.0 to 1.0
- Lower values = more detections (but possibly more false positives)
- Default: 0.3

## Algorithm Details

1. **Detection**: GroundingDINO detects all instances matching the text prompt
2. **Filtering**: Removes low-confidence detections below box_threshold
3. **Merging**: Groups detections with IoU > iou_threshold and creates union bounding boxes
4. **Expansion**: Expands final boxes by bezel_expansion factor to include screen edges
5. **Cropping**: Extracts each screen region as a separate image
6. **Visualization**: Creates annotated image with final merged boxes

## Troubleshooting

### No screens detected
- Try lowering `--box_threshold` (e.g., 0.2 or 0.1)
- Check if the text prompt matches what you want to detect

### Too many small detections
- Increase `--iou_threshold` for more aggressive merging
- Increase `--box_threshold` to filter out weak detections

### Cropped images miss screen edges  
- Increase `--bezel_expansion` (e.g., 0.1 for 10% expansion)

### Multiple boxes on same screen
- Lower `--iou_threshold` for more aggressive merging
- The algorithm uses IoU to determine if boxes should be merged

## Output Files

For each run, the following files are created in the output directory:

1. **original_image.jpg** - Copy of input image
2. **detected_screens.jpg** - Visualization with bounding boxes
3. **screen_crop_N.jpg** - Individual cropped screens (N = 1, 2, 3, ...)

Each cropped image contains a single screen with expanded boundaries to include the bezel area.
