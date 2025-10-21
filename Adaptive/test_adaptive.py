import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_RGB import get_test_data
from Adaptive.model_adaptive import AdaptiveMultiscaleNet
from utils.image_utils import save_img
import torchvision.transforms as transforms


def calculate_psnr(img1, img2, max_value=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(max_value / torch.sqrt(mse))


def calculate_ssim(img1, img2, max_value=1.0):
    """Calculate SSIM between two images"""
    def gaussian_window(size, sigma):
        coords = torch.arange(size).float()
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(1) * g.unsqueeze(0)
    
    window = gaussian_window(11, 1.5).unsqueeze(0).unsqueeze(0)
    if img1.is_cuda:
        window = window.cuda()
    
    mu1 = F.conv2d(img1, window, padding=5, groups=1)
    mu2 = F.conv2d(img2, window, padding=5, groups=1)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=5, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=5, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=5, groups=1) - mu1_mu2
    
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def analyze_adaptive_sampling(model, input_img, save_path=None):
    """
    Analyze and visualize adaptive sampling patterns
    """
    model.eval()
    with torch.no_grad():
        # Get contrast maps from adaptive INR modules
        if hasattr(model, 'INR') and hasattr(model.INR, 'contrast_detector'):
            # Small scale contrast
            inp_img_small = F.interpolate(input_img, scale_factor=0.25)
            contrast_small = model.INR.contrast_detector(inp_img_small)
            density_small = model.INR.sampler.map_density(contrast_small)
            
            # Mid scale contrast
            inp_img_mid = F.interpolate(input_img, scale_factor=0.5)
            contrast_mid = model.INR2.contrast_detector(inp_img_mid)
            density_mid = model.INR2.sampler.map_density(contrast_mid)
            
            # Convert to numpy for visualization
            contrast_small_np = contrast_small[0, 0].cpu().numpy()
            density_small_np = density_small[0, 0].cpu().numpy()
            contrast_mid_np = contrast_mid[0, 0].cpu().numpy()
            density_mid_np = density_mid[0, 0].cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original images
            orig_small = inp_img_small[0].permute(1, 2, 0).cpu().numpy()
            orig_mid = inp_img_mid[0].permute(1, 2, 0).cpu().numpy()
            orig_full = input_img[0].permute(1, 2, 0).cpu().numpy()
            
            axes[0, 0].imshow(orig_small)
            axes[0, 0].set_title('Input (Small Scale)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(contrast_small_np, cmap='viridis')
            axes[0, 1].set_title('Contrast Map (Small)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(density_small_np, cmap='plasma')
            axes[0, 2].set_title('Sampling Density (Small)')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(orig_mid)
            axes[1, 0].set_title('Input (Mid Scale)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(contrast_mid_np, cmap='viridis')
            axes[1, 1].set_title('Contrast Map (Mid)')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(density_mid_np, cmap='plasma')
            axes[1, 2].set_title('Sampling Density (Mid)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            # Return statistics
            stats = {
                'small_scale': {
                    'contrast_mean': float(contrast_small_np.mean()),
                    'contrast_std': float(contrast_small_np.std()),
                    'density_mean': float(density_small_np.mean()),
                    'density_std': float(density_small_np.std()),
                },
                'mid_scale': {
                    'contrast_mean': float(contrast_mid_np.mean()),
                    'contrast_std': float(contrast_mid_np.std()),
                    'density_mean': float(density_mid_np.mean()),
                    'density_std': float(density_mid_np.std()),
                }
            }
            
            return stats
    
    return None


def test_model(model, test_loader, device, save_results=False, results_dir=None, analyze_sampling=False):
    """Test the model and compute metrics"""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0
    num_images = 0
    
    if save_results and results_dir:
        os.makedirs(results_dir, exist_ok=True)
        if analyze_sampling:
            os.makedirs(os.path.join(results_dir, 'sampling_analysis'), exist_ok=True)
    
    sampling_stats = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for batch_idx, (input_imgs, target_imgs, filenames) in enumerate(progress_bar):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(input_imgs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            
            restored_imgs = outputs[0]  # Main output
            
            # Calculate metrics for each image in batch
            for i in range(input_imgs.size(0)):
                restored = restored_imgs[i:i+1]
                target = target_imgs[i:i+1]
                
                # Clamp values to [0, 1]
                restored = torch.clamp(restored, 0, 1)
                target = torch.clamp(target, 0, 1)
                
                # Calculate PSNR and SSIM
                psnr = calculate_psnr(restored, target)
                ssim = calculate_ssim(restored, target)
                
                total_psnr += psnr.item()
                total_ssim += ssim.item()
                total_time += inference_time / input_imgs.size(0)
                num_images += 1
                
                # Update progress bar
                avg_psnr = total_psnr / num_images
                avg_ssim = total_ssim / num_images
                avg_time = total_time / num_images
                
                progress_bar.set_postfix({
                    'PSNR': f'{avg_psnr:.2f}dB',
                    'SSIM': f'{avg_ssim:.4f}',
                    'Time': f'{avg_time*1000:.1f}ms'
                })
                
                # Save results if requested
                if save_results and results_dir:
                    filename = filenames[i] if isinstance(filenames, list) else f"result_{batch_idx}_{i}"
                    
                    # Save restored image
                    save_path = os.path.join(results_dir, f"{filename}_restored.png")
                    save_img(restored, save_path)
                    
                    # Save comparison
                    comparison = torch.cat([input_imgs[i:i+1], restored, target], dim=3)
                    comp_path = os.path.join(results_dir, f"{filename}_comparison.png")
                    save_img(comparison, comp_path)
                    
                    # Analyze adaptive sampling if requested
                    if analyze_sampling and hasattr(model, 'INR'):
                        sampling_path = os.path.join(results_dir, 'sampling_analysis', f"{filename}_sampling.png")
                        stats = analyze_adaptive_sampling(model, input_imgs[i:i+1], sampling_path)
                        if stats:
                            stats['filename'] = filename
                            stats['psnr'] = psnr.item()
                            stats['ssim'] = ssim.item()
                            sampling_stats.append(stats)
    
    # Calculate final averages
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_time = total_time / num_images
    
    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_time_ms': avg_time * 1000,
        'num_images': num_images,
        'sampling_stats': sampling_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Test Adaptive NeRD-Rain Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='./Datasets', help='Path to test dataset')
    parser.add_argument('--results_dir', type=str, default='./test_results_adaptive', help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=48, help='Model dimension')
    parser.add_argument('--use_adaptive_inr', action='store_true', default=True, help='Use adaptive INR')
    parser.add_argument('--contrast_method', type=str, default='combined', 
                       choices=['sobel', 'laplacian', 'gradient', 'combined'],
                       help='Contrast detection method')
    parser.add_argument('--min_density', type=float, default=0.1, help='Minimum sampling density')
    parser.add_argument('--max_density', type=float, default=1.0, help='Maximum sampling density')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--save_results', action='store_true', help='Save test results')
    parser.add_argument('--analyze_sampling', action='store_true', help='Analyze adaptive sampling patterns')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = AdaptiveMultiscaleNet(
        dim=args.dim,
        use_adaptive_inr=args.use_adaptive_inr,
        contrast_method=args.contrast_method,
        min_density=args.min_density,
        max_density=args.max_density
    ).to(device)
    
    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (legacy format)")
    
    # Create test dataset
    test_dataset = get_test_data(args.data_path, {})
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Adaptive INR: {args.use_adaptive_inr}")
    print(f"Contrast method: {args.contrast_method}")
    print(f"Sampling density range: [{args.min_density}, {args.max_density}]")
    
    # Test the model
    print("Starting testing...")
    results = test_model(
        model, test_loader, device, 
        save_results=args.save_results,
        results_dir=args.results_dir,
        analyze_sampling=args.analyze_sampling
    )
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Average PSNR: {results['avg_psnr']:.3f} dB")
    print(f"Average SSIM: {results['avg_ssim']:.4f}")
    print(f"Average inference time: {results['avg_time_ms']:.1f} ms")
    print(f"Number of test images: {results['num_images']}")
    
    # Save detailed results
    if args.save_results:
        import json
        
        results_file = os.path.join(args.results_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {results_file}")
        
        # Save sampling statistics if available
        if results['sampling_stats']:
            stats_file = os.path.join(args.results_dir, 'sampling_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(results['sampling_stats'], f, indent=2)
            print(f"Sampling statistics saved to: {stats_file}")
            
            # Print sampling analysis summary
            print("\nAdaptive Sampling Analysis:")
            print("-" * 30)
            
            small_contrasts = [s['small_scale']['contrast_mean'] for s in results['sampling_stats']]
            small_densities = [s['small_scale']['density_mean'] for s in results['sampling_stats']]
            mid_contrasts = [s['mid_scale']['contrast_mean'] for s in results['sampling_stats']]
            mid_densities = [s['mid_scale']['density_mean'] for s in results['sampling_stats']]
            
            print(f"Small scale - Avg contrast: {np.mean(small_contrasts):.4f} ± {np.std(small_contrasts):.4f}")
            print(f"Small scale - Avg density: {np.mean(small_densities):.4f} ± {np.std(small_densities):.4f}")
            print(f"Mid scale - Avg contrast: {np.mean(mid_contrasts):.4f} ± {np.std(mid_contrasts):.4f}")
            print(f"Mid scale - Avg density: {np.mean(mid_densities):.4f} ± {np.std(mid_densities):.4f}")


if __name__ == '__main__':
    main()