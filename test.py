import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from model import MultiscaleNet as mynet
#from model_S import MultiscaleNet as myNet
from skimage import img_as_ubyte
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = argparse.ArgumentParser(description='Image Deraining')
parser.add_argument('--input_dir', default='/kaggle/input/gtav-nightrain-rerendered-version/test/rainy/', type=str, help='Directory of rainy images')
parser.add_argument('--gt_dir', default='/kaggle/input/gtav-nightrain-rerendered-version/test/gt/', type=str, help='Directory of ground truth images')
parser.add_argument('--output_dir', default='./results/GTAV_NightRain', type=str, help='Directory to save results')
parser.add_argument('--weights', default='', type=str, help='Path to weights') 
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--win_size', default=256, type=int, help='window size')
parser.add_argument('--save_images', action='store_true', help='Save derained images')
args = parser.parse_args()

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
    """Calculate SSIM between two images"""
    # Convert to Y channel if RGB
    img1_y = rgb_to_y(img1.astype(np.float64))
    img2_y = rgb_to_y(img2.astype(np.float64))
    
    # Convert to uint8 for SSIM calculation
    img1_y = np.clip(img1_y, 0, 255).astype(np.uint8)
    img2_y = np.clip(img2_y, 0, 255).astype(np.uint8)
    
    try:
        # Use skimage ssim if available
        from skimage.metrics import structural_similarity
        ssim_val = structural_similarity(img1_y, img2_y, data_range=255)
    except ImportError:
        # Fallback to manual SSIM calculation
        ssim_val = compute_ssim_manual(img1_y, img2_y)
    
    return ssim_val

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
    base_name = rainy_filename.split('_')[0]
    ext = os.path.splitext(rainy_filename)[1]
    gt_filename = base_name + ext
    return gt_filename
result_dir = args.output_dir
gt_dir = args.gt_dir
win = args.win_size
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model_restoration = mynet()
get_parameter_number(model_restoration)
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Custom dataset loader for GTAV NightRain dataset
rgb_dir_test = args.input_dir
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

utils.mkdir(result_dir)

# Create directory for metrics logging
metrics_dir = os.path.join(result_dir, 'metrics')
utils.mkdir(metrics_dir)

with torch.no_grad():
    psnr_list = []
    ssim_list = []
    results_log = []
    
    print(f"Testing on dataset: {rgb_dir_test}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Results will be saved to: {result_dir}")
    
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = data_test[0].cuda()
        filenames = data_test[1]
        _, _, Hx, Wx = input_.shape
        filenames = data_test[1]
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
                utils.save_img((os.path.join(result_dir, filename + '.png')), restored_img_uint8)
            
            # Get corresponding ground truth image
            gt_filename = get_corresponding_gt(filename + '.png')
            gt_path = os.path.join(gt_dir, gt_filename)
            
            if os.path.exists(gt_path):
                # Load ground truth image
                try:
                    import cv2
                    gt_img = cv2.imread(gt_path)
                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                except ImportError:
                    # Fallback to PIL if cv2 not available
                    from PIL import Image
                    gt_img = Image.open(gt_path)
                    gt_img = np.array(gt_img)
                
                # Ensure both images have the same dimensions
                if gt_img.shape[:2] != restored_img_uint8.shape[:2]:
                    print(f"Warning: Size mismatch for {filename}")
                    print(f"  Restored: {restored_img_uint8.shape}")
                    print(f"  GT: {gt_img.shape}")
                    # Resize restored image to match GT
                    try:
                        import cv2
                        restored_img_uint8 = cv2.resize(restored_img_uint8, (gt_img.shape[1], gt_img.shape[0]))
                    except ImportError:
                        from PIL import Image
                        restored_pil = Image.fromarray(restored_img_uint8)
                        restored_pil = restored_pil.resize((gt_img.shape[1], gt_img.shape[0]), Image.LANCZOS)
                        restored_img_uint8 = np.array(restored_pil)
                
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
                
                # Print progress every 50 images
                if (ii + 1) % 50 == 0:
                    avg_psnr = np.mean(psnr_list)
                    avg_ssim = np.mean(ssim_list)
                    print(f"\nProgress: {ii+1} images processed")
                    print(f"Current averages - PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}")
            else:
                print(f"Warning: Ground truth image not found: {gt_path}")
    
    # Calculate final metrics
    if psnr_list and ssim_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        std_psnr = np.std(psnr_list)
        std_ssim = np.std(ssim_list)
        
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"Number of images processed: {len(psnr_list)}")
        print(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print(f"PSNR range: [{np.min(psnr_list):.4f}, {np.max(psnr_list):.4f}]")
        print(f"SSIM range: [{np.min(ssim_list):.4f}, {np.max(ssim_list):.4f}]")
        
        # Save detailed results to file
        results_file = os.path.join(metrics_dir, 'detailed_results.txt')
        with open(results_file, 'w') as f:
            f.write("Image-wise Results:\n")
            f.write("=" * 50 + "\n")
            for result in results_log:
                f.write(f"{result['filename']} -> {result['gt_filename']}: ")
                f.write(f"PSNR={result['psnr']:.4f}, SSIM={result['ssim']:.4f}\n")
            
            f.write(f"\nSummary Statistics:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of images: {len(psnr_list)}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
            f.write(f"PSNR range: [{np.min(psnr_list):.4f}, {np.max(psnr_list):.4f}]\n")
            f.write(f"SSIM range: [{np.min(ssim_list):.4f}, {np.max(ssim_list):.4f}]\n")
        
        # Save metrics as CSV for further analysis
        import csv
        csv_file = os.path.join(metrics_dir, 'results.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'gt_filename', 'psnr', 'ssim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results_log:
                writer.writerow(result)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"CSV results saved to: {csv_file}")
    else:
        print("\nNo valid image pairs found for evaluation!")
        print("Please check your input and ground truth directories.")
