import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss

class SSIMLoss(nn.Module):
    """SSIM Loss for structural similarity"""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        img1 = img1.to('cuda:0')
        img2 = img2.to('cuda:0')
        
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        ssim_value = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1 - ssim_value  # Convert to loss (lower SSIM = higher loss)

class IlluminationAwareLoss(nn.Module):
    """Illumination-aware loss for maintaining consistent lighting"""
    
    def __init__(self, eps=1e-8):
        super(IlluminationAwareLoss, self).__init__()
        self.eps = eps
        
    def _rgb_to_luminance(self, img):
        """Convert RGB to luminance using standard weights"""
        # Using ITU-R BT.709 standard weights
        weights = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        if img.is_cuda:
            weights = weights.cuda(img.get_device())
        weights = weights.type_as(img)
        
        luminance = torch.sum(img * weights, dim=1, keepdim=True)
        return luminance
    
    def _compute_local_mean(self, img, kernel_size=5):
        """Compute local mean using average pooling"""
        padding = kernel_size // 2
        avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        return avg_pool(img)
    
    def forward(self, restored, target):
        restored = restored.to('cuda:0')
        target = target.to('cuda:0')
        
        # Convert to luminance
        restored_lum = self._rgb_to_luminance(restored)
        target_lum = self._rgb_to_luminance(target)
        
        # Compute local illumination statistics
        restored_local_mean = self._compute_local_mean(restored_lum)
        target_local_mean = self._compute_local_mean(target_lum)
        
        # Illumination consistency loss
        illum_loss = F.l1_loss(restored_local_mean, target_local_mean)
        
        # Global illumination loss
        restored_global_mean = torch.mean(restored_lum, dim=[2, 3], keepdim=True)
        target_global_mean = torch.mean(target_lum, dim=[2, 3], keepdim=True)
        global_illum_loss = F.l1_loss(restored_global_mean, target_global_mean)
        
        # Combine local and global illumination losses
        total_loss = illum_loss + 0.5 * global_illum_loss
        
        return total_loss
