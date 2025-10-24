# GTAV NightRain Dataset Testing Guide

## Overview
This guide explains how to test your deraining model on the GTAV NightRain dataset with automatic PSNR and SSIM calculation.

## Dataset Structure
The GTAV NightRain dataset has a specific structure:
```
/kaggle/input/gtav-nightrain-rerendered-version/test/
├── gt/           # Ground truth images
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
└── rainy/        # Rainy images (6 per GT)
    ├── 0000_00.png
    ├── 0000_01.png
    ├── 0000_02.png
    ├── 0000_03.png
    ├── 0000_04.png
    ├── 0000_05.png
    ├── 0001_00.png
    └── ...
```

## Files Modified/Created

### 1. `test.py` (Original - Updated)
- Added PSNR and SSIM calculation functions
- Modified to handle the GT-rainy mapping (0000.png ↔ 0000_XX.png)
- Added comprehensive metric logging

### 2. `test_gtav.py` (New - Recommended)
- Standalone script specifically for GTAV dataset
- Better error handling and fallback methods
- Doesn't depend on external CV2/skimage for core functionality
- More robust image loading with PIL/OpenCV fallbacks

## Usage Examples

### Basic Testing (Save metrics only)
```bash
python test_gtav.py --input_dir /kaggle/input/gtav-nightrain-rerendered-version/test/rainy/ \
                    --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/test/gt/ \
                    --output_dir ./results/GTAV_NightRain \
                    --weights path/to/your/model.pth
```

### Full Testing (Save images + metrics)
```bash
python test_gtav.py --input_dir /kaggle/input/gtav-nightrain-rerendered-version/test/rainy/ \
                    --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/test/gt/ \
                    --output_dir ./results/GTAV_NightRain \
                    --weights path/to/your/model.pth \
                    --save_images
```

### With Custom Parameters
```bash
python test_gtav.py --input_dir /kaggle/input/gtav-nightrain-rerendered-version/test/rainy/ \
                    --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/test/gt/ \
                    --output_dir ./results/GTAV_NightRain \
                    --weights path/to/your/model.pth \
                    --win_size 512 \
                    --batch_size 2 \
                    --save_images
```

## Output Structure
After running the test, you'll get:
```
results/GTAV_NightRain/
├── metrics/
│   ├── detailed_results.txt    # Per-image PSNR/SSIM
│   ├── results.csv            # CSV format for analysis
│   └── summary.txt            # Final averages
└── *.png                      # Derained images (if --save_images)
```

## Key Features

### 1. Automatic GT Matching
- Automatically maps rainy images to GT: `0000_XX.png` → `0000.png`
- Handles missing GT files gracefully
- Reports mapping issues

### 2. Robust Metric Calculation
- PSNR calculation using Y channel (luminance)
- SSIM calculation with manual implementation (no external deps)
- Handles image size mismatches with automatic resizing

### 3. Comprehensive Logging
- Progress tracking every 100 images
- Detailed per-image results
- Statistical summary (mean, std, min, max)
- Multiple output formats (TXT, CSV)

### 4. Error Handling
- Validates all input paths before starting
- Graceful handling of missing/corrupted images
- Fallback image loading methods (PIL → OpenCV)
- Memory management with CUDA cache clearing

## Performance Tips

### Memory Optimization
- Use smaller batch sizes for large images
- Adjust `win_size` based on GPU memory
- The script automatically clears CUDA cache

### Speed Optimization
- Use `--batch_size > 1` if memory allows
- Don't use `--save_images` if you only need metrics
- Use SSD storage for faster I/O

## Expected Results Format

### Console Output
```
Testing on dataset: /kaggle/input/gtav-nightrain-rerendered-version/test/rainy/
Ground truth directory: /kaggle/input/gtav-nightrain-rerendered-version/test/gt/
Results will be saved to: ./results/GTAV_NightRain
Total test samples: 1200

Progress: 600 images processed
Current averages - PSNR: 28.45 dB, SSIM: 0.8234

============================================================
FINAL RESULTS:
============================================================
Number of images processed: 1200
Average PSNR: 28.67 ± 2.34 dB
Average SSIM: 0.8245 ± 0.0456
PSNR range: [22.13, 35.89]
SSIM range: [0.7234, 0.9123]
```

### CSV Output (results.csv)
```csv
filename,gt_filename,psnr,ssim
0000_00,0000.png,28.45,0.8234
0000_01,0000.png,27.89,0.8156
...
```

## Troubleshooting

### Common Issues
1. **Import errors**: The script has fallbacks for missing libraries
2. **Path issues**: Always use absolute paths in Kaggle
3. **Memory errors**: Reduce batch_size or win_size
4. **Missing GT**: Check the filename mapping logic

### Dependencies
Minimal requirements:
- PyTorch
- NumPy
- PIL (Pillow) or OpenCV (fallback)
- tqdm

Optional:
- scikit-image (for advanced SSIM)
- OpenCV (for image operations)

## Integration with Existing Code
The `test_gtav.py` script is standalone and doesn't require modifications to your existing training code. It uses the same model loading utilities and maintains compatibility with your current architecture.