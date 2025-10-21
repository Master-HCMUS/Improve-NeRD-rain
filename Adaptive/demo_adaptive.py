import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Adaptive.model_adaptive import AdaptiveMultiscaleNet
from Adaptive.adaptive_inr import ContrastDetector, AdaptiveSampler
from utils.image_utils import save_img


def load_image(image_path, device):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor, image.size


def analyze_contrast_methods(image_tensor, save_dir):
    """Analyze different contrast detection methods"""
    os.makedirs(save_dir, exist_ok=True)
    
    methods = ['sobel', 'laplacian', 'gradient', 'combined']
    contrast_maps = {}
    
    for method in methods:
        detector = ContrastDetector(method=method).to(image_tensor.device)
        contrast_map = detector(image_tensor)
        contrast_maps[method] = contrast_map[0, 0].cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    orig_img = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Contrast maps
    for i, (method, contrast_map) in enumerate(contrast_maps.items()):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        im = axes[row, col].imshow(contrast_map, cmap='viridis')
        axes[row, col].set_title(f'Contrast Map ({method.capitalize()})')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col])
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'contrast_methods_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return contrast_maps


def analyze_sampling_densities(image_tensor, contrast_map, save_dir):
    """Analyze different sampling density configurations"""
    os.makedirs(save_dir, exist_ok=True)
    
    density_configs = [
        (0.1, 1.0),  # Default
        (0.05, 1.0), # More aggressive
        (0.2, 0.8),  # More conservative
        (0.3, 0.7),  # Very conservative
    ]
    
    density_maps = {}
    
    for min_density, max_density in density_configs:
        sampler = AdaptiveSampler(min_density=min_density, max_density=max_density).to(image_tensor.device)
        contrast_tensor = torch.from_numpy(contrast_map).unsqueeze(0).unsqueeze(0).to(image_tensor.device)
        density_map = sampler.map_density(contrast_tensor)
        density_maps[f'{min_density}-{max_density}'] = density_map[0, 0].cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    orig_img = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Contrast map
    axes[0, 1].imshow(contrast_map, cmap='viridis')
    axes[0, 1].set_title('Contrast Map')
    axes[0, 1].axis('off')
    
    # Density maps
    for i, (config, density_map) in enumerate(density_maps.items()):
        if i < 4:
            row = i // 2
            col = (i % 2) + 1 if row == 0 else i % 2
            if row == 0 and col == 2:
                row = 1
                col = 0
            
            im = axes[row, col].imshow(density_map, cmap='plasma')
            axes[row, col].set_title(f'Density Map ({config})')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col])
    
    # Remove empty subplots
    if len(density_maps) < 4:
        for i in range(len(density_maps), 4):
            row = 1
            col = i - 2
            if col < 3:
                axes[row, col].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sampling_densities_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return density_maps


def demo_adaptive_inference(model, image_tensor, save_dir):
    """Demonstrate adaptive inference with analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        start_time = time.time()
        outputs = model(image_tensor)
        inference_time = time.time() - start_time
        
        restored_image = outputs[0]  # Main output
        
        # Get intermediate outputs for visualization
        if len(outputs) > 1:
            small_output = outputs[-1]  # Small scale output
            mid_output = outputs[-2] if len(outputs) > 2 else None
        
        # Analyze adaptive sampling at different scales
        sampling_analysis = {}
        
        # Small scale analysis
        if hasattr(model, 'INR') and hasattr(model.INR, 'contrast_detector'):
            inp_img_small = F.interpolate(image_tensor, scale_factor=0.25)
            contrast_small = model.INR.contrast_detector(inp_img_small)
            density_small = model.INR.sampler.map_density(contrast_small)
            
            sampling_analysis['small'] = {
                'input': inp_img_small[0].permute(1, 2, 0).cpu().numpy(),
                'contrast': contrast_small[0, 0].cpu().numpy(),
                'density': density_small[0, 0].cpu().numpy(),
                'output': small_output[0].permute(1, 2, 0).cpu().numpy() if small_output is not None else None
            }
        
        # Mid scale analysis
        if hasattr(model, 'INR2') and hasattr(model.INR2, 'contrast_detector'):
            inp_img_mid = F.interpolate(image_tensor, scale_factor=0.5)
            contrast_mid = model.INR2.contrast_detector(inp_img_mid)
            density_mid = model.INR2.sampler.map_density(contrast_mid)
            
            sampling_analysis['mid'] = {
                'input': inp_img_mid[0].permute(1, 2, 0).cpu().numpy(),
                'contrast': contrast_mid[0, 0].cpu().numpy(),
                'density': density_mid[0, 0].cpu().numpy(),
                'output': mid_output[0].permute(1, 2, 0).cpu().numpy() if mid_output is not None else None
            }
        
        # Create comprehensive visualization
        create_inference_visualization(
            image_tensor, restored_image, sampling_analysis, 
            inference_time, save_dir
        )
        
        return {
            'inference_time': inference_time,
            'restored_image': restored_image,
            'sampling_analysis': sampling_analysis
        }


def create_inference_visualization(input_tensor, output_tensor, sampling_analysis, inference_time, save_dir):
    """Create comprehensive visualization of adaptive inference"""
    
    # Main comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original and result
    input_img = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    output_img = torch.clamp(output_tensor[0], 0, 1).permute(1, 2, 0).cpu().numpy()
    
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(output_img)
    axes[0, 1].set_title(f'Restored Image\n(Time: {inference_time*1000:.1f}ms)')
    axes[0, 1].axis('off')
    
    # Small scale analysis
    if 'small' in sampling_analysis:
        small_data = sampling_analysis['small']
        
        axes[0, 2].imshow(small_data['input'])
        axes[0, 2].set_title('Small Scale Input')
        axes[0, 2].axis('off')
        
        im1 = axes[0, 3].imshow(small_data['contrast'], cmap='viridis')
        axes[0, 3].set_title('Small Scale Contrast')
        axes[0, 3].axis('off')
        plt.colorbar(im1, ax=axes[0, 3])
        
        im2 = axes[1, 0].imshow(small_data['density'], cmap='plasma')
        axes[1, 0].set_title('Small Scale Density')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])
        
        if small_data['output'] is not None:
            axes[1, 1].imshow(small_data['output'])
            axes[1, 1].set_title('Small Scale Output')
            axes[1, 1].axis('off')
    
    # Mid scale analysis
    if 'mid' in sampling_analysis:
        mid_data = sampling_analysis['mid']
        
        im3 = axes[1, 2].imshow(mid_data['contrast'], cmap='viridis')
        axes[1, 2].set_title('Mid Scale Contrast')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2])
        
        im4 = axes[1, 3].imshow(mid_data['density'], cmap='plasma')
        axes[1, 3].set_title('Mid Scale Density')
        axes[1, 3].axis('off')
        plt.colorbar(im4, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adaptive_inference_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual results
    save_img(input_tensor, os.path.join(save_dir, 'input.png'))
    save_img(torch.clamp(output_tensor, 0, 1), os.path.join(save_dir, 'output.png'))


def main():
    parser = argparse.ArgumentParser(description='Adaptive NeRD-Rain Demo')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to adaptive model checkpoint')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./demo_results', help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=48, help='Model dimension')
    parser.add_argument('--contrast_method', type=str, default='combined', 
                       choices=['sobel', 'laplacian', 'gradient', 'combined'],
                       help='Contrast detection method')
    parser.add_argument('--min_density', type=float, default=0.1, help='Minimum sampling density')
    parser.add_argument('--max_density', type=float, default=1.0, help='Maximum sampling density')
    
    # Demo options
    parser.add_argument('--analyze_contrast', action='store_true', help='Analyze different contrast methods')
    parser.add_argument('--analyze_density', action='store_true', help='Analyze different density configurations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load image
    if not os.path.isfile(args.input_image):
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    
    print(f"Loading image: {args.input_image}")
    image_tensor, original_size = load_image(args.input_image, device)
    print(f"Image size: {original_size}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = AdaptiveMultiscaleNet(
        dim=args.dim,
        use_adaptive_inr=True,
        contrast_method=args.contrast_method,
        min_density=args.min_density,
        max_density=args.max_density
    ).to(device)
    
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (legacy format)")
    
    print(f"Model configuration:")
    print(f"  Contrast method: {args.contrast_method}")
    print(f"  Sampling density range: [{args.min_density}, {args.max_density}]")
    
    # Run adaptive inference demo
    print("\nRunning adaptive inference...")
    results = demo_adaptive_inference(model, image_tensor, args.output_dir)
    
    print(f"Inference completed in {results['inference_time']*1000:.1f} ms")
    
    # Analyze contrast methods if requested
    if args.analyze_contrast:
        print("\nAnalyzing contrast detection methods...")
        contrast_maps = analyze_contrast_methods(
            image_tensor, 
            os.path.join(args.output_dir, 'contrast_analysis')
        )
        
        # Print contrast statistics
        print("Contrast detection statistics:")
        for method, contrast_map in contrast_maps.items():
            print(f"  {method.capitalize()}: mean={contrast_map.mean():.4f}, std={contrast_map.std():.4f}")
    
    # Analyze sampling densities if requested
    if args.analyze_density:
        print("\nAnalyzing sampling density configurations...")
        # Use the contrast map from the selected method
        detector = ContrastDetector(method=args.contrast_method).to(device)
        contrast_map = detector(image_tensor)[0, 0].cpu().numpy()
        
        density_maps = analyze_sampling_densities(
            image_tensor, 
            contrast_map,
            os.path.join(args.output_dir, 'density_analysis')
        )
        
        # Print density statistics
        print("Sampling density statistics:")
        for config, density_map in density_maps.items():
            print(f"  {config}: mean={density_map.mean():.4f}, std={density_map.std():.4f}")
    
    # Print sampling analysis if available
    if results['sampling_analysis']:
        print("\nAdaptive sampling analysis:")
        for scale, data in results['sampling_analysis'].items():
            print(f"  {scale.capitalize()} scale:")
            print(f"    Contrast: mean={data['contrast'].mean():.4f}, std={data['contrast'].std():.4f}")
            print(f"    Density: mean={data['density'].mean():.4f}, std={data['density'].std():.4f}")
    
    print(f"\nDemo completed! Results saved to: {args.output_dir}")
    print("Files generated:")
    print("  - input.png: Original input image")
    print("  - output.png: Restored image")
    print("  - adaptive_inference_analysis.png: Comprehensive analysis visualization")
    
    if args.analyze_contrast:
        print("  - contrast_analysis/: Contrast detection method comparison")
    
    if args.analyze_density:
        print("  - density_analysis/: Sampling density configuration comparison")


if __name__ == '__main__':
    main()