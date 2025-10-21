import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_RGB import get_test_data
from Adaptive.model_adaptive import AdaptiveMultiscaleNet
from model import MultiscaleNet  # Original model
from utils.image_utils import save_img


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


def load_model(checkpoint_path, model_class, device, **model_kwargs):
    """Load a model from checkpoint"""
    model = model_class(**model_kwargs).to(device)
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate_model(model, test_loader, device, model_name="Model"):
    """Evaluate a model on test dataset"""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0
    num_images = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Evaluating {model_name}")
        
        for batch_idx, (input_imgs, target_imgs, _) in enumerate(progress_bar):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(input_imgs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            
            # Get the main output
            if isinstance(outputs, list):
                restored_imgs = outputs[0]  # Main output for multiscale models
            else:
                restored_imgs = outputs
            
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
    
    return {
        'avg_psnr': total_psnr / num_images,
        'avg_ssim': total_ssim / num_images,
        'avg_time_ms': (total_time / num_images) * 1000,
        'num_images': num_images
    }


def compare_on_sample_images(baseline_model, adaptive_model, test_loader, device, save_dir, num_samples=5):
    """Compare models on sample images and save results"""
    baseline_model.eval()
    adaptive_model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    sample_results = []
    
    with torch.no_grad():
        for batch_idx, (input_imgs, target_imgs, filenames) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
                
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Get outputs from both models
            baseline_outputs = baseline_model(input_imgs)
            adaptive_outputs = adaptive_model(input_imgs)
            
            baseline_restored = baseline_outputs[0] if isinstance(baseline_outputs, list) else baseline_outputs
            adaptive_restored = adaptive_outputs[0] if isinstance(adaptive_outputs, list) else adaptive_outputs
            
            # Process each image in batch
            for i in range(input_imgs.size(0)):
                input_img = input_imgs[i:i+1]
                target_img = target_imgs[i:i+1]
                baseline_img = torch.clamp(baseline_restored[i:i+1], 0, 1)
                adaptive_img = torch.clamp(adaptive_restored[i:i+1], 0, 1)
                
                # Calculate metrics
                baseline_psnr = calculate_psnr(baseline_img, target_img)
                adaptive_psnr = calculate_psnr(adaptive_img, target_img)
                baseline_ssim = calculate_ssim(baseline_img, target_img)
                adaptive_ssim = calculate_ssim(adaptive_img, target_img)
                
                # Create comparison image
                comparison = torch.cat([
                    input_img, 
                    baseline_img, 
                    adaptive_img, 
                    target_img
                ], dim=3)
                
                # Save comparison
                filename = filenames[i] if isinstance(filenames, list) else f"sample_{batch_idx}_{i}"
                save_path = os.path.join(save_dir, f"{filename}_comparison.png")
                save_img(comparison, save_path)
                
                # Store results
                sample_results.append({
                    'filename': filename,
                    'baseline_psnr': baseline_psnr.item(),
                    'adaptive_psnr': adaptive_psnr.item(),
                    'baseline_ssim': baseline_ssim.item(),
                    'adaptive_ssim': adaptive_ssim.item(),
                    'psnr_improvement': adaptive_psnr.item() - baseline_psnr.item(),
                    'ssim_improvement': adaptive_ssim.item() - baseline_ssim.item()
                })
    
    return sample_results


def create_comparison_plots(baseline_results, adaptive_results, sample_results, save_dir):
    """Create comparison plots and save them"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Overall comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR comparison
    models = ['Baseline', 'Adaptive']
    psnr_values = [baseline_results['avg_psnr'], adaptive_results['avg_psnr']]
    bars1 = ax1.bar(models, psnr_values, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Average PSNR Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, psnr_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom')
    
    # SSIM comparison
    ssim_values = [baseline_results['avg_ssim'], adaptive_results['avg_ssim']]
    bars2 = ax2.bar(models, ssim_values, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('SSIM')
    ax2.set_title('Average SSIM Comparison')
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, ssim_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Inference time comparison
    time_values = [baseline_results['avg_time_ms'], adaptive_results['avg_time_ms']]
    bars3 = ax3.bar(models, time_values, color=['skyblue', 'lightcoral'])
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Average Inference Time')
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, time_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Sample-wise improvements
    if sample_results:
        psnr_improvements = [r['psnr_improvement'] for r in sample_results]
        ssim_improvements = [r['ssim_improvement'] for r in sample_results]
        sample_names = [f"S{i+1}" for i in range(len(sample_results))]
        
        x = np.arange(len(sample_names))
        width = 0.35
        
        bars4a = ax4.bar(x - width/2, psnr_improvements, width, label='PSNR Improvement', alpha=0.8)
        ax4_twin = ax4.twinx()
        bars4b = ax4_twin.bar(x + width/2, ssim_improvements, width, label='SSIM Improvement', 
                             color='orange', alpha=0.8)
        
        ax4.set_xlabel('Sample Images')
        ax4.set_ylabel('PSNR Improvement (dB)')
        ax4_twin.set_ylabel('SSIM Improvement')
        ax4.set_title('Per-Sample Improvements')
        ax4.set_xticks(x)
        ax4.set_xticklabels(sample_names)
        ax4.grid(True, alpha=0.3)
        
        # Add legends
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create improvement histogram if sample results available
    if sample_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        psnr_improvements = [r['psnr_improvement'] for r in sample_results]
        ssim_improvements = [r['ssim_improvement'] for r in sample_results]
        
        ax1.hist(psnr_improvements, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('PSNR Improvement (dB)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of PSNR Improvements')
        ax1.axvline(np.mean(psnr_improvements), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(psnr_improvements):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(ssim_improvements, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('SSIM Improvement')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of SSIM Improvements')
        ax2.axvline(np.mean(ssim_improvements), color='red', linestyle='--',
                   label=f'Mean: {np.mean(ssim_improvements):.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'improvement_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs Adaptive NeRD-Rain Models')
    parser.add_argument('--baseline_checkpoint', type=str, required=True, help='Path to baseline model checkpoint')
    parser.add_argument('--adaptive_checkpoint', type=str, required=True, help='Path to adaptive model checkpoint')
    parser.add_argument('--data_path', type=str, default='./Datasets', help='Path to test dataset')
    parser.add_argument('--results_dir', type=str, default='./comparison_results', help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=48, help='Model dimension')
    parser.add_argument('--contrast_method', type=str, default='combined', 
                       choices=['sobel', 'laplacian', 'gradient', 'combined'],
                       help='Contrast detection method for adaptive model')
    parser.add_argument('--min_density', type=float, default=0.1, help='Minimum sampling density')
    parser.add_argument('--max_density', type=float, default=1.0, help='Maximum sampling density')
    
    # Comparison parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of sample images to compare')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading baseline model...")
    baseline_model = load_model(
        args.baseline_checkpoint, 
        MultiscaleNet, 
        device,
        dim=args.dim
    )
    
    print("Loading adaptive model...")
    adaptive_model = load_model(
        args.adaptive_checkpoint,
        AdaptiveMultiscaleNet,
        device,
        dim=args.dim,
        use_adaptive_inr=True,
        contrast_method=args.contrast_method,
        min_density=args.min_density,
        max_density=args.max_density
    )
    
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
    print(f"Adaptive contrast method: {args.contrast_method}")
    print(f"Sampling density range: [{args.min_density}, {args.max_density}]")
    
    # Evaluate both models
    print("\nEvaluating models...")
    baseline_results = evaluate_model(baseline_model, test_loader, device, "Baseline")
    adaptive_results = evaluate_model(adaptive_model, test_loader, device, "Adaptive")
    
    # Compare on sample images
    print(f"\nComparing models on {args.num_samples} sample images...")
    sample_results = compare_on_sample_images(
        baseline_model, adaptive_model, test_loader, device,
        os.path.join(args.results_dir, 'sample_comparisons'),
        args.num_samples
    )
    
    # Print comparison results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print("\nBaseline Model:")
    print(f"  Average PSNR: {baseline_results['avg_psnr']:.3f} dB")
    print(f"  Average SSIM: {baseline_results['avg_ssim']:.4f}")
    print(f"  Average inference time: {baseline_results['avg_time_ms']:.1f} ms")
    
    print("\nAdaptive Model:")
    print(f"  Average PSNR: {adaptive_results['avg_psnr']:.3f} dB")
    print(f"  Average SSIM: {adaptive_results['avg_ssim']:.4f}")
    print(f"  Average inference time: {adaptive_results['avg_time_ms']:.1f} ms")
    
    # Calculate improvements
    psnr_improvement = adaptive_results['avg_psnr'] - baseline_results['avg_psnr']
    ssim_improvement = adaptive_results['avg_ssim'] - baseline_results['avg_ssim']
    time_change = adaptive_results['avg_time_ms'] - baseline_results['avg_time_ms']
    
    print("\nImprovements:")
    print(f"  PSNR: {psnr_improvement:+.3f} dB ({psnr_improvement/baseline_results['avg_psnr']*100:+.2f}%)")
    print(f"  SSIM: {ssim_improvement:+.4f} ({ssim_improvement/baseline_results['avg_ssim']*100:+.2f}%)")
    print(f"  Time: {time_change:+.1f} ms ({time_change/baseline_results['avg_time_ms']*100:+.2f}%)")
    
    # Sample-wise analysis
    if sample_results:
        sample_psnr_improvements = [r['psnr_improvement'] for r in sample_results]
        sample_ssim_improvements = [r['ssim_improvement'] for r in sample_results]
        
        print(f"\nSample-wise Analysis ({len(sample_results)} samples):")
        print(f"  PSNR improvements: {np.mean(sample_psnr_improvements):.3f} ± {np.std(sample_psnr_improvements):.3f} dB")
        print(f"  SSIM improvements: {np.mean(sample_ssim_improvements):.4f} ± {np.std(sample_ssim_improvements):.4f}")
        print(f"  Best PSNR improvement: {max(sample_psnr_improvements):.3f} dB")
        print(f"  Best SSIM improvement: {max(sample_ssim_improvements):.4f}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(
        baseline_results, adaptive_results, sample_results,
        args.results_dir
    )
    
    # Save detailed results
    comparison_data = {
        'baseline_results': baseline_results,
        'adaptive_results': adaptive_results,
        'improvements': {
            'psnr_improvement': float(psnr_improvement),
            'ssim_improvement': float(ssim_improvement),
            'time_change_ms': float(time_change),
            'psnr_improvement_percent': float(psnr_improvement/baseline_results['avg_psnr']*100),
            'ssim_improvement_percent': float(ssim_improvement/baseline_results['avg_ssim']*100),
            'time_change_percent': float(time_change/baseline_results['avg_time_ms']*100)
        },
        'sample_results': sample_results,
        'config': {
            'contrast_method': args.contrast_method,
            'min_density': args.min_density,
            'max_density': args.max_density,
            'baseline_checkpoint': args.baseline_checkpoint,
            'adaptive_checkpoint': args.adaptive_checkpoint
        }
    }
    
    results_file = os.path.join(args.results_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    print(f"Comparison plots saved to: {args.results_dir}")
    print("Comparison completed!")


if __name__ == '__main__':
    main()