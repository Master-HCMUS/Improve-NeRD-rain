#!/usr/bin/env python3
"""
Test script for GTAV NightRain dataset with PSNR/SSIM evaluation
Modified for the specific dataset structure where each GT image has 6 corresponding rainy images.

Usage:
python test.py --input_dir /kaggle/input/gtav-nightrain-rerendered-version/test/rainy/ \
               --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/test/gt/ \
               --output_dir ./results/GTAV_NightRain \
               --weights path/to/your/model.pth \
               --save_images

Dataset structure expected:
- GT: 0000.png, 0001.png, 0002.png, ...
- Rainy: 0000_00.png, 0000_01.png, ..., 0000_05.png, 0001_00.png, ...
"""

import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from model import MultiscaleNet as mynet
from skimage import img_as_ubyte
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *
import numpy as np

def rgb_to_y(img):
    """Convert RGB to Y channel (luminance)"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return img

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Convert to Y channel if RGB
    img1_y = rgb_to_y(img1.astype(np.float64))
    img2_y = rgb_to_y(img2.astype(np.float64))
    
    mse = np.mean((img1_y - img2_y) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images using manual implementation"""
    # Convert to Y channel if RGB
    img1_y = rgb_to_y(img1.astype(np.float64))
    img2_y = rgb_to_y(img2.astype(np.float64))
    
    # Convert to uint8 for SSIM calculation
    img1_y = np.clip(img1_y, 0, 255).astype(np.uint8)
    img2_y = np.clip(img2_y, 0, 255).astype(np.uint8)
    
    return compute_ssim_manual(img1_y, img2_y)

def compute_ssim_manual(img1, img2):
    """Manual SSIM calculation"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim_val = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_val

def get_corresponding_gt(rainy_filename):
    """Get corresponding ground truth filename for a rainy image"""
    # Extract base name (remove _XX suffix)
    # Example: 0000_00.png -> 0000.png
    parts = rainy_filename.split('_')
    if len(parts) >= 2:
        base_name = parts[0]
        ext = os.path.splitext(rainy_filename)[1]
        gt_filename = base_name + ext
        return gt_filename
    else:
        # If no underscore pattern, return as is
        return rainy_filename

def load_image(image_path):
    """Load image with fallback methods"""
    try:
        # Try PIL first (more commonly available)
        from PIL import Image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except ImportError:
        try:
            # Fallback to cv2
            import cv2
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except ImportError:
            raise ImportError("Neither PIL nor OpenCV is available. Please install one of them.")

def resize_image(img, target_size):
    """Resize image with fallback methods"""
    try:
        # Try PIL first
        from PIL import Image
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
        return np.array(img_pil)
    except ImportError:
        try:
            # Fallback to cv2
            import cv2
            return cv2.resize(img, (target_size[1], target_size[0]))
        except ImportError:
            raise ImportError("Neither PIL nor OpenCV is available for resizing.")

def main():
    parser = argparse.ArgumentParser(description='Image Deraining with PSNR/SSIM Evaluation')
    parser.add_argument('--input_dir', default='/kaggle/input/gtav-nightrain-rerendered-version/test/rainy/', 
                       type=str, help='Directory of rainy images')
    parser.add_argument('--gt_dir', default='/kaggle/input/gtav-nightrain-rerendered-version/test/gt/', 
                       type=str, help='Directory of ground truth images')
    parser.add_argument('--output_dir', default='./results/GTAV_NightRain', 
                       type=str, help='Directory to save results')
    parser.add_argument('--weights', required=True, type=str, help='Path to model weights') 
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--win_size', default=256, type=int, help='window size')
    parser.add_argument('--save_images', action='store_true', help='Save derained images')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory does not exist: {args.gt_dir}")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights file does not exist: {args.weights}")
        return
    
    # Setup directories
    result_dir = args.output_dir
    gt_dir = args.gt_dir
    win = args.win_size
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Load model
    print("Loading model...")
    model_restoration = mynet()
    get_parameter_number(model_restoration)
    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    
    # Setup data loader
    rgb_dir_test = args.input_dir
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    
    utils.mkdir(result_dir)
    metrics_dir = os.path.join(result_dir, 'metrics')
    utils.mkdir(metrics_dir)
    
    print(f"Testing on dataset: {rgb_dir_test}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Results will be saved to: {result_dir}")
    print(f"Total test samples: {len(test_dataset)}")
    
    # Testing loop
    with torch.no_grad():
        psnr_list = []
        ssim_list = []
        results_log = []
        processed_count = 0
        
        for ii, data_test in enumerate(tqdm(test_loader, desc="Processing images")):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            input_ = data_test[0].cuda()
            filenames = data_test[1]
            _, _, Hx, Wx = input_.shape
            
            # Process through model
            input_re, batch_list = window_partitionx(input_, win)
            restored = model_restoration(input_re)
            restored = window_reversex(restored[0], win, Hx, Wx, batch_list)
            
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            
            for batch in range(len(restored)):
                restored_img = restored[batch]
                filename = filenames[batch]
                
                # Convert restored image to uint8
                restored_img_uint8 = img_as_ubyte(restored_img)
                
                # Save derained image if requested
                if args.save_images:
                    output_path = os.path.join(result_dir, filename + '.png')
                    utils.save_img(output_path, restored_img_uint8)
                
                # Get corresponding ground truth image
                gt_filename = get_corresponding_gt(filename + '.png')
                gt_path = os.path.join(gt_dir, gt_filename)
                
                if os.path.exists(gt_path):
                    try:
                        # Load ground truth image
                        gt_img = load_image(gt_path)
                        
                        # Ensure both images have the same dimensions
                        if gt_img.shape[:2] != restored_img_uint8.shape[:2]:
                            print(f"\\nResizing for {filename}: GT{gt_img.shape} -> Restored{restored_img_uint8.shape}")
                            restored_img_uint8 = resize_image(restored_img_uint8, gt_img.shape[:2])
                        
                        # Calculate PSNR and SSIM
                        psnr_val = calculate_psnr(restored_img_uint8, gt_img)
                        ssim_val = calculate_ssim(restored_img_uint8, gt_img)
                        
                        psnr_list.append(psnr_val)
                        ssim_list.append(ssim_val)
                        
                        # Log results
                        result_entry = {
                            'filename': filename,
                            'gt_filename': gt_filename,
                            'psnr': psnr_val,
                            'ssim': ssim_val
                        }
                        results_log.append(result_entry)
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"\\nError processing {filename}: {str(e)}")
                        continue
                else:
                    print(f"\\nWarning: Ground truth not found: {gt_path}")
            
            # Print progress every 100 images
            if (ii + 1) % 100 == 0 and psnr_list:
                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)
                print(f"\\nProgress: {processed_count} images processed")
                print(f"Current averages - PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")
    
    # Calculate and display final results
    if psnr_list and ssim_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        std_psnr = np.std(psnr_list)
        std_ssim = np.std(ssim_list)
        
        print("\\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"Number of images processed: {len(psnr_list)}")
        print(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print(f"PSNR range: [{np.min(psnr_list):.4f}, {np.max(psnr_list):.4f}]")
        print(f"SSIM range: [{np.min(ssim_list):.4f}, {np.max(ssim_list):.4f}]")
        
        # Save detailed results
        results_file = os.path.join(metrics_dir, 'detailed_results.txt')
        with open(results_file, 'w') as f:
            f.write("GTAV NightRain Dataset Evaluation Results\\n")
            f.write("="*50 + "\\n")
            f.write(f"Model: {args.weights}\\n")
            f.write(f"Input Directory: {args.input_dir}\\n")
            f.write(f"GT Directory: {args.gt_dir}\\n")
            f.write(f"Number of images: {len(psnr_list)}\\n\\n")
            
            f.write("Image-wise Results:\\n")
            f.write("-" * 50 + "\\n")
            for result in results_log:
                f.write(f"{result['filename']} -> {result['gt_filename']}: ")
                f.write(f"PSNR={result['psnr']:.4f}, SSIM={result['ssim']:.4f}\\n")
            
            f.write(f"\\nSummary Statistics:\\n")
            f.write("-" * 50 + "\\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB\\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\\n")
            f.write(f"PSNR range: [{np.min(psnr_list):.4f}, {np.max(psnr_list):.4f}]\\n")
            f.write(f"SSIM range: [{np.min(ssim_list):.4f}, {np.max(ssim_list):.4f}]\\n")
        
        # Save CSV for analysis
        import csv
        csv_file = os.path.join(metrics_dir, 'results.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'gt_filename', 'psnr', 'ssim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results_log:
                writer.writerow(result)
        
        print(f"\\nDetailed results saved to: {results_file}")
        print(f"CSV results saved to: {csv_file}")
        
        # Save summary metrics
        summary_file = os.path.join(metrics_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"PSNR: {avg_psnr:.4f}\\n")
            f.write(f"SSIM: {avg_ssim:.4f}\\n")
        
        print(f"Summary saved to: {summary_file}")
        
    else:
        print("\\nNo valid image pairs found for evaluation!")
        print("Please check your input and ground truth directories.")
        print("Expected structure:")
        print("  GT: 0000.png, 0001.png, ...")
        print("  Rainy: 0000_00.png, 0000_01.png, ..., 0000_05.png, ...")

if __name__ == '__main__':
    main()