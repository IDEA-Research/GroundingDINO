# Quick Start Guide: Batch Screen Processing

## 🚀 Three Ways to Process Your Medical Images

### 1. Single Image (Original functionality)
```bash
python screen_crop_demo.py -i your_image.jpg -o output_folder
```

### 2. Batch Processing (Full featured)
```bash
python screen_crop_demo.py -f /path/to/images/ -o output_folder
```

### 3. Simple Batch Processing (Streamlined)
```bash
python batch_screen_crop.py input_folder output_folder
```

## 📁 What You Get

Each method creates organized output with:
```
output_folder/
├── image1_name/
│   ├── original_image.jpg     (or original.jpg for simple batch)
│   ├── detected_screens.jpg   (or detection.jpg for simple batch)
│   ├── screen_crop_1.jpg
│   ├── screen_crop_2.jpg
│   └── screen_crop_N.jpg
├── image2_name/
│   └── ...
```

## ⚙️ Common Options

- `--box_threshold 0.25` - Lower = more sensitive detection
- `--iou_threshold 0.2` - Lower = more aggressive box merging  
- `--bezel_expansion 0.08` - Higher = larger cropped area around screens
- `--cpu-only` - Use CPU instead of GPU

## 🔧 Troubleshooting

**No screens detected?**
```bash
python screen_crop_demo.py -f your_folder/ -o output --box_threshold 0.2
```

**Too many small detections?**
```bash
python screen_crop_demo.py -f your_folder/ -o output --iou_threshold 0.2
```

**Cropped screens cut off edges?**
```bash
python screen_crop_demo.py -f your_folder/ -o output --bezel_expansion 0.1
```

## 📊 Example Output

For your medical frame, you'll typically get:
- 7 individual screen crops per image
- Clean visualization with merged bounding boxes
- Original image preserved
- Automatic folder organization by image name

The system automatically merges overlapping detections and expands bounding boxes to capture complete screen bezels.
