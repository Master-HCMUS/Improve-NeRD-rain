import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import save_image
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
from tqdm import tqdm
import time
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_RGB import get_training_data, get_validation_data
from Adaptive.model_adaptive import AdaptiveMultiscaleNet
from losses import CharbonnierLoss, EdgeLoss, fftLoss, SSIMLoss, IlluminationAwareLoss


def network_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AdaptiveSamplingLoss(nn.Module):
    """
    Loss component that encourages better adaptive sampling
    """
    def __init__(self, alpha=0.1):
        super(AdaptiveSamplingLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predicted, target, contrast_map=None):
        """
        Compute adaptive sampling loss
        
        Args:
            predicted: predicted output
            target: ground truth
            contrast_map: contrast map from adaptive INR (optional)
        """
        base_loss = self.mse_loss(predicted, target)
        
        if contrast_map is not None:
            # Encourage higher accuracy in high-contrast regions
            contrast_weighted_loss = torch.mean(contrast_map * (predicted - target) ** 2)
            return base_loss + self.alpha * contrast_weighted_loss
        
        return base_loss


def train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    
    # Initialize loss components
    charbonnier_loss = CharbonnierLoss().to(device)
    edge_loss = EdgeLoss().to(device)
    fft_loss = fftLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    illum_loss = IlluminationAwareLoss().to(device)
    adaptive_loss = AdaptiveSamplingLoss().to(device)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (target_imgs, input_imgs, filename) in enumerate(progress_bar):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_imgs)
        final_output = outputs[0]  # Main output
        
        # Compute multi-scale losses
        total_loss = 0
        for i, output in enumerate(outputs):
            if i < len(outputs) - 1:  # Intermediate outputs
                target_resized = F.interpolate(target_imgs, size=output.shape[-2:])
            else:  # Final output
                target_resized = target_imgs
            
            # Basic reconstruction loss
            char_loss = charbonnier_loss(output, target_resized)
            
            # Perceptual losses
            edge_loss_val = edge_loss(output, target_resized)
            fft_loss_val = fft_loss(output, target_resized)
            ssim_loss_val = ssim_loss(output, target_resized)
            illum_loss_val = illum_loss(output, target_resized)
            
            # Adaptive sampling loss (only for final output)
            if i == 0 and hasattr(model, 'INR') and hasattr(model.INR, 'contrast_detector'):
                # Get contrast map from the adaptive INR
                with torch.no_grad():
                    contrast_map = model.INR.contrast_detector(input_imgs)
                adapt_loss = adaptive_loss(output, target_resized, contrast_map)
            else:
                adapt_loss = adaptive_loss(output, target_resized)
            
            # Combine losses
            scale_loss = (char_loss + 
                         0.1 * edge_loss_val + 
                         0.1 * fft_loss_val + 
                         0.1 * ssim_loss_val + 
                         0.05 * illum_loss_val +
                         0.1 * adapt_loss)
            
            # Weight intermediate outputs less
            weight = 1.0 if i == 0 else 0.5
            total_loss += weight * scale_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Avg': f'{running_loss/(batch_idx+1):.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % 100 == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Training/Loss', total_loss.item(), global_step)
            writer.add_scalar('Training/Charbonnier', char_loss.item(), global_step)
            writer.add_scalar('Training/Edge', edge_loss_val.item(), global_step)
            writer.add_scalar('Training/FFT', fft_loss_val.item(), global_step)
            writer.add_scalar('Training/SSIM', ssim_loss_val.item(), global_step)
            writer.add_scalar('Training/Illumination', illum_loss_val.item(), global_step)
            writer.add_scalar('Training/Adaptive', adapt_loss.item(), global_step)
    
    return running_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, writer, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    num_batches = len(val_loader)
    
    charbonnier_loss = CharbonnierLoss().to(device)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation")
        
        for batch_idx, (target_imgs, input_imgs, filename) in enumerate(progress_bar):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Forward pass
            outputs = model(input_imgs)
            final_output = outputs[0]  # Main output
            
            # Compute loss
            loss = charbonnier_loss(final_output, target_imgs)
            running_loss += loss.item()
            
            # Compute PSNR
            mse = F.mse_loss(final_output, target_imgs)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            running_psnr += psnr.item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PSNR': f'{psnr.item():.2f}dB'
            })
    
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    
    # Log to tensorboard
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/PSNR', avg_psnr, epoch)
    
    return avg_loss, avg_psnr


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filename)


def main():
    parser = argparse.ArgumentParser(description='Train Adaptive NeRD-Rain Model')
    parser.add_argument('--data_path', type=str, default='./Datasets', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_adaptive', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs_adaptive', help='Directory for tensorboard logs')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=48, help='Model dimension')
    parser.add_argument('--use_adaptive_inr', action='store_true', default=True, help='Use adaptive INR')
    parser.add_argument('--contrast_method', type=str, default='combined', 
                       choices=['sobel', 'laplacian', 'gradient', 'combined'],
                       help='Contrast detection method')
    parser.add_argument('--min_density', type=float, default=0.1, help='Minimum sampling density')
    parser.add_argument('--max_density', type=float, default=1.0, help='Maximum sampling density')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patch_size', type=int, default=256, help='Training patch size')
    
    # Scheduler parameters
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--scheduler_type', type=str, default='cosine', 
                       choices=['cosine', 'step', 'exponential'], help='Scheduler type')
    
    # Checkpoint parameters
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Validate data path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please check if the dataset directory exists or provide correct path with --data_path")
        print("\nExpected dataset structure:")
        print("  Datasets/")
        print("    Rain200L/")
        print("      train/")
        print("        input/")
        print("        target/")
        print("      test/")
        print("        input/")
        print("        target/")
        return
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data path: {os.path.abspath(args.data_path)}")
    
    # Create model
    model = AdaptiveMultiscaleNet(
        dim=args.dim,
        use_adaptive_inr=args.use_adaptive_inr,
        contrast_method=args.contrast_method,
        min_density=args.min_density,
        max_density=args.max_density
    ).to(device)
    
    print(f"Model parameters: {network_parameters(model)}")
    
    # Create datasets and data loaders
    train_dataset = get_training_data(args.data_path, {'patch_size': args.patch_size})
    val_dataset = get_validation_data(args.data_path, {'patch_size': args.patch_size})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Setup scheduler
    if args.scheduler_type == 'cosine':
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs - args.warmup_epochs)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_cosine)
    elif args.scheduler_type == 'step':
        scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_step)
    else:
        scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_exp)
    
    # Setup loss criterion
    criterion = CharbonnierLoss().to(device)
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resumed from epoch {start_epoch}")
    
    print("Starting training...")
    print(f"Adaptive INR: {args.use_adaptive_inr}")
    print(f"Contrast method: {args.contrast_method}")
    print(f"Sampling density range: [{args.min_density}, {args.max_density}]")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        
        # Validate
        val_loss, val_psnr = validate_epoch(model, val_loader, criterion, device, writer, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val PSNR: {val_psnr:.2f} dB")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.save_dir, 'best_model.pth')
            )
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.num_epochs-1, val_loss,
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    writer.close()
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.6f}")


if __name__ == '__main__':
    main()