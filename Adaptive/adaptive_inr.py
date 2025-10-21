import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import MLP, make_coord, L


class ContrastDetector(nn.Module):
    """
    Module to compute local contrast using multiple edge detection methods
    """
    def __init__(self, method='combined'):
        super(ContrastDetector, self).__init__()
        self.method = method
        
        # Sobel kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Laplacian kernel
        self.laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        
        # Register as buffers to move with the model
        self.register_buffer('sobel_x_kernel', self.sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y_kernel', self.sobel_y.view(1, 1, 3, 3))
        self.register_buffer('laplacian_kernel', self.laplacian.view(1, 1, 3, 3))
        
    def rgb_to_gray(self, img):
        """Convert RGB to grayscale using luminance weights"""
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        weights = weights.to(img.device).type_as(img)
        return torch.sum(img * weights, dim=1, keepdim=True)
    
    def sobel_edges(self, gray_img):
        """Compute Sobel edge magnitude"""
        # Apply Sobel filters
        grad_x = F.conv2d(gray_img, self.sobel_x_kernel, padding=1)
        grad_y = F.conv2d(gray_img, self.sobel_y_kernel, padding=1)
        
        # Compute magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return magnitude
    
    def laplacian_edges(self, gray_img):
        """Compute Laplacian edge response"""
        edges = F.conv2d(gray_img, self.laplacian_kernel, padding=1)
        return torch.abs(edges)
    
    def gradient_magnitude(self, gray_img):
        """Compute gradient magnitude using central differences"""
        # Horizontal gradient
        grad_x = gray_img[:, :, :, 2:] - gray_img[:, :, :, :-2]
        grad_x = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')
        
        # Vertical gradient
        grad_y = gray_img[:, :, 2:, :] - gray_img[:, :, :-2, :]
        grad_y = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')
        
        # Magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return magnitude
    
    def forward(self, img):
        """
        Compute contrast map from input image
        Args:
            img: Input image tensor [B, C, H, W]
        Returns:
            contrast_map: Contrast map [B, 1, H, W]
        """
        # Convert to grayscale
        gray_img = self.rgb_to_gray(img)
        
        if self.method == 'sobel':
            contrast = self.sobel_edges(gray_img)
        elif self.method == 'laplacian':
            contrast = self.laplacian_edges(gray_img)
        elif self.method == 'gradient':
            contrast = self.gradient_magnitude(gray_img)
        elif self.method == 'combined':
            # Combine multiple edge detection methods
            sobel_contrast = self.sobel_edges(gray_img)
            laplacian_contrast = self.laplacian_edges(gray_img)
            gradient_contrast = self.gradient_magnitude(gray_img)
            
            # Weighted combination
            contrast = 0.5 * sobel_contrast + 0.3 * laplacian_contrast + 0.2 * gradient_contrast
        else:
            raise ValueError(f"Unknown contrast detection method: {self.method}")
        
        return contrast


class AdaptiveSampler(nn.Module):
    """
    Module to generate adaptive sampling grids based on contrast maps
    """
    def __init__(self, 
                 min_density=0.1, 
                 max_density=1.0, 
                 density_levels=5,
                 smoothing_kernel_size=5):
        super(AdaptiveSampler, self).__init__()
        self.min_density = min_density
        self.max_density = max_density
        self.density_levels = density_levels
        self.smoothing_kernel_size = smoothing_kernel_size
        
        # Gaussian smoothing kernel for contrast map
        sigma = smoothing_kernel_size / 6.0
        kernel_1d = torch.exp(-0.5 * (torch.arange(smoothing_kernel_size) - smoothing_kernel_size//2)**2 / sigma**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer('smoothing_kernel', kernel_2d.view(1, 1, smoothing_kernel_size, smoothing_kernel_size))
    
    def normalize_contrast_map(self, contrast_map):
        """
        Normalize contrast map to [0, 1] range
        """
        batch_size = contrast_map.shape[0]
        normalized_maps = []
        
        for i in range(batch_size):
            single_map = contrast_map[i:i+1]
            min_val = single_map.min()
            max_val = single_map.max()
            
            if max_val > min_val:
                normalized_map = (single_map - min_val) / (max_val - min_val)
            else:
                normalized_map = torch.zeros_like(single_map)
            
            normalized_maps.append(normalized_map)
        
        return torch.cat(normalized_maps, dim=0)
    
    def smooth_contrast_map(self, contrast_map):
        """Apply Gaussian smoothing to contrast map"""
        padding = self.smoothing_kernel_size // 2
        smoothed = F.conv2d(contrast_map, self.smoothing_kernel, padding=padding)
        return smoothed
    
    def map_contrast_to_density(self, normalized_contrast):
        """
        Map normalized contrast values to sampling densities
        """
        # Use exponential mapping to emphasize high contrast regions
        density_map = self.min_density + (self.max_density - self.min_density) * (normalized_contrast ** 0.5)
        return density_map
    
    def generate_adaptive_coordinates(self, density_map, target_height, target_width):
        """
        Generate adaptive sampling coordinates based on density map
        """
        batch_size = density_map.shape[0]
        device = density_map.device
        
        all_coords = []
        all_weights = []
        
        for b in range(batch_size):
            single_density = density_map[b, 0]  # [H, W]
            
            # Create base coordinate grid
            y_coords = torch.linspace(-1, 1, target_height, device=device)
            x_coords = torch.linspace(-1, 1, target_width, device=device)
            
            # Adaptive sampling based on density
            adaptive_coords = []
            sampling_weights = []
            
            for i in range(target_height):
                for j in range(target_width):
                    # Get density at this location
                    h_idx = int(i * single_density.shape[0] / target_height)
                    w_idx = int(j * single_density.shape[1] / target_width)
                    h_idx = min(h_idx, single_density.shape[0] - 1)
                    w_idx = min(w_idx, single_density.shape[1] - 1)
                    
                    local_density = single_density[h_idx, w_idx].item()
                    
                    # Sample multiple points in high-density regions
                    if local_density > 0.7:  # High density
                        num_samples = 4
                    elif local_density > 0.4:  # Medium density
                        num_samples = 2
                    else:  # Low density
                        num_samples = 1
                    
                    # Generate samples around the current position
                    base_y, base_x = y_coords[i], x_coords[j]
                    cell_size_y = 2.0 / target_height
                    cell_size_x = 2.0 / target_width
                    
                    for s in range(num_samples):
                        if num_samples == 1:
                            sample_y, sample_x = base_y, base_x
                        else:
                            # Add small random offsets within the cell
                            offset_y = (torch.rand(1, device=device) - 0.5) * cell_size_y * 0.8
                            offset_x = (torch.rand(1, device=device) - 0.5) * cell_size_x * 0.8
                            sample_y = base_y + offset_y
                            sample_x = base_x + offset_x
                        
                        adaptive_coords.append([sample_y.item(), sample_x.item()])
                        sampling_weights.append(local_density / num_samples)
            
            coords_tensor = torch.tensor(adaptive_coords, device=device).float()
            weights_tensor = torch.tensor(sampling_weights, device=device).float()
            
            all_coords.append(coords_tensor)
            all_weights.append(weights_tensor)
        
        return all_coords, all_weights
    
    def forward(self, contrast_map, target_height, target_width):
        """
        Generate adaptive sampling grid
        Args:
            contrast_map: Contrast map [B, 1, H, W]
            target_height: Target height for sampling
            target_width: Target width for sampling
        Returns:
            coords_list: List of coordinate tensors for each batch
            weights_list: List of sampling weights for each batch
            density_map: Processed density map [B, 1, H, W]
        """
        # Normalize contrast map
        normalized_contrast = self.normalize_contrast_map(contrast_map)
        
        # Apply smoothing
        smoothed_contrast = self.smooth_contrast_map(normalized_contrast)
        
        # Map to density
        density_map = self.map_contrast_to_density(smoothed_contrast)
        
        # Generate adaptive coordinates
        coords_list, weights_list = self.generate_adaptive_coordinates(
            density_map, target_height, target_width
        )
        
        return coords_list, weights_list, density_map


class AdaptiveINR(nn.Module):
    """
    Adaptive Implicit Neural Representation with contrast-based sampling
    """
    def __init__(self, dim, local_ensemble=True, feat_unfold=True, cell_decode=True,
                 contrast_method='combined', min_density=0.1, max_density=1.0):
        super(AdaptiveINR, self).__init__()
        
        # Initialize base INR components
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        # Initialize adaptive components
        self.contrast_detector = ContrastDetector(method=contrast_method)
        self.adaptive_sampler = AdaptiveSampler(
            min_density=min_density, 
            max_density=max_density
        )
        
        # MLP network for INR
        imnet_in_dim = dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 + 4 * L  # coordinates + positional encoding
        if self.cell_decode:
            imnet_in_dim += 2
            
        self.imnet = MLP(imnet_in_dim, 3, [256, 256, 256])
    
    def positional_encoding(self, input, L): 
        """Positional encoding for coordinates"""
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi  
        spectrum = input[..., None] * freq  
        sin, cos = spectrum.sin(), spectrum.cos()  
        input_enc = torch.stack([sin, cos], dim=-2)  
        input_enc = input_enc.view(*shape[:-1], -1)  
        return input_enc
    
    def query_rgb_adaptive(self, inp, coords_list, weights_list, cell_list=None):
        """
        Query RGB values using adaptive sampling
        """
        feat = inp
        
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        batch_size = feat.shape[0]
        device = feat.device
        
        # Create feature coordinate grid
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        
        all_outputs = []
        
        for b in range(batch_size):
            coords = coords_list[b]  # [N, 2]
            weights = weights_list[b]  # [N]
            
            if cell_list is not None:
                cell = cell_list[b]
            else:
                # Default cell size
                cell = torch.ones_like(coords)
                cell[:, 0] *= 2 / feat.shape[-2]
                cell[:, 1] *= 2 / feat.shape[-1]
            
            # Add positional encoding to coordinates
            coords_enc = self.positional_encoding(coords.unsqueeze(0), L=L).squeeze(0)
            coords_with_enc = torch.cat([coords, coords_enc], dim=-1)
            
            # Get feature coordinates for interpolation
            bs, q, h, w = feat[b:b+1].shape
            q_feat = feat[b:b+1].view(1, q, -1).permute(0, 2, 1)  # [1, H*W, C]
            
            bs, q, h, w = feat_coord[b:b+1].shape
            q_coord = feat_coord[b:b+1].view(1, q, -1).permute(0, 2, 1)  # [1, H*W, 2]
            
            # Add positional encoding to feature coordinates
            points_enc = self.positional_encoding(q_coord, L=L)
            q_coord_enc = torch.cat([q_coord, points_enc], dim=-1)  # [1, H*W, 2+4*L]
            
            # For each query coordinate, find relative position to feature coordinates
            num_queries = coords.shape[0]
            query_results = []
            
            for q_idx in range(num_queries):
                query_coord = coords_with_enc[q_idx:q_idx+1].unsqueeze(0)  # [1, 1, 2+4*L]
                
                # Compute relative coordinates
                rel_coord = query_coord[:, :, :2] - q_coord[:, :, :2]  # [1, H*W, 2]
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # Combine features
                inp_mlp = torch.cat([q_feat, rel_coord], dim=-1)  # [1, H*W, C+2]
                
                if self.cell_decode:
                    query_cell = cell[q_idx:q_idx+1].unsqueeze(0)  # [1, 1, 2]
                    rel_cell = query_cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    rel_cell_expanded = rel_cell.expand(-1, q_feat.shape[1], -1)  # [1, H*W, 2]
                    inp_mlp = torch.cat([inp_mlp, rel_cell_expanded], dim=-1)
                
                # Query the MLP
                pred = self.imnet(inp_mlp.view(-1, inp_mlp.shape[-1])).view(1, -1, 3)  # [1, H*W, 3]
                
                # Weighted average based on spatial proximity (simple bilinear-like weighting)
                coord_2d = query_coord[:, :, :2]  # [1, 1, 2]
                feat_coord_2d = q_coord[:, :, :2]  # [1, H*W, 2]
                
                distances = torch.norm(coord_2d - feat_coord_2d, dim=-1, keepdim=True)  # [1, H*W, 1]
                spatial_weights = torch.exp(-distances * 10)  # Exponential weighting
                spatial_weights = spatial_weights / (spatial_weights.sum(dim=1, keepdim=True) + 1e-8)
                
                weighted_pred = (pred * spatial_weights).sum(dim=1)  # [1, 3]
                query_results.append(weighted_pred)
            
            # Combine results for this batch
            batch_output = torch.cat(query_results, dim=0)  # [N, 3]
            all_outputs.append(batch_output)
        
        return all_outputs
    
    def reconstruct_image(self, query_results, coords_list, weights_list, output_height, output_width):
        """
        Reconstruct full image from adaptive query results
        """
        batch_size = len(query_results)
        device = query_results[0].device
        
        reconstructed_images = []
        
        for b in range(batch_size):
            results = query_results[b]  # [N, 3]
            coords = coords_list[b]  # [N, 2]
            weights = weights_list[b]  # [N]
            
            # Create output image
            output_img = torch.zeros(3, output_height, output_width, device=device)
            weight_map = torch.zeros(1, output_height, output_width, device=device)
            
            # Map coordinates to pixel locations
            pixel_coords = coords.clone()
            pixel_coords[:, 0] = (pixel_coords[:, 0] + 1) * (output_height - 1) / 2
            pixel_coords[:, 1] = (pixel_coords[:, 1] + 1) * (output_width - 1) / 2
            
            # Bilinear splatting
            for i in range(len(results)):
                y, x = pixel_coords[i]
                value = results[i]  # [3]
                weight = weights[i]
                
                # Get integer coordinates
                y0, x0 = int(torch.floor(y)), int(torch.floor(x))
                y1, x1 = y0 + 1, x0 + 1
                
                # Clamp to image bounds
                y0, y1 = max(0, y0), min(output_height - 1, y1)
                x0, x1 = max(0, x0), min(output_width - 1, x1)
                
                if y0 < output_height and x0 < output_width:
                    # Bilinear weights
                    wy1, wy0 = y - y0, y1 - y
                    wx1, wx0 = x - x0, x1 - x
                    
                    # Distribute value to four nearest pixels
                    if y0 < output_height and x0 < output_width:
                        w = wy0 * wx0 * weight
                        output_img[:, y0, x0] += w * value
                        weight_map[0, y0, x0] += w
                    
                    if y0 < output_height and x1 < output_width:
                        w = wy0 * wx1 * weight
                        output_img[:, y0, x1] += w * value
                        weight_map[0, y0, x1] += w
                    
                    if y1 < output_height and x0 < output_width:
                        w = wy1 * wx0 * weight
                        output_img[:, y1, x0] += w * value
                        weight_map[0, y1, x0] += w
                    
                    if y1 < output_height and x1 < output_width:
                        w = wy1 * wx1 * weight
                        output_img[:, y1, x1] += w * value
                        weight_map[0, y1, x1] += w
            
            # Normalize by weights
            output_img = output_img / (weight_map + 1e-8)
            reconstructed_images.append(output_img.unsqueeze(0))
        
        return torch.cat(reconstructed_images, dim=0)
    
    def forward(self, inp, input_img=None):
        """
        Forward pass with adaptive query resolution
        Args:
            inp: Feature tensor from encoder [B, C, H, W]
            input_img: Original input image for contrast computation [B, 3, H, W]
        Returns:
            output: Reconstructed image [B, 3, H, W]
        """
        batch_size, _, feat_h, feat_w = inp.shape
        
        # Use input image for contrast if provided, otherwise use features
        if input_img is not None:
            contrast_input = input_img
        else:
            # Convert features to 3-channel for contrast computation
            if inp.shape[1] >= 3:
                contrast_input = inp[:, :3, :, :]
            else:
                contrast_input = inp.repeat(1, 3, 1, 1)
        
        # Step 1: Compute contrast map
        contrast_map = self.contrast_detector(contrast_input)
        
        # Step 2: Generate adaptive sampling grid
        coords_list, weights_list, density_map = self.adaptive_sampler(
            contrast_map, feat_h * 2, feat_w * 2  # Upsample by 2x
        )
        
        # Step 3: Query INR with adaptive grid
        query_results = self.query_rgb_adaptive(inp, coords_list, weights_list)
        
        # Step 4: Reconstruct image
        output = self.reconstruct_image(
            query_results, coords_list, weights_list, 
            feat_h * 2, feat_w * 2  # Output size
        )
        
        return output