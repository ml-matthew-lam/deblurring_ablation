import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd

class LossCombination(nn.Module): 
    def __init__(self, alpha, window_size=11, sigma=1.5):
        '''
        Params:
            window_size (int): should be an odd integer
        '''
        super().__init__()
        self.alpha = alpha 
        self.window_size = window_size
        self.l1 = nn.L1Loss()

        coordinates = torch.arange(start=-(window_size//2), end=(window_size//2) + 1).float()
        gaussian = torch.exp(-(coordinates/sigma)**2 / 2) 
        
        # create the 2D window
        window = torch.ger(gaussian, gaussian)
        window = window / window.sum()

        # expand to 3 channels + force physical memory allocation
        window = window.unsqueeze(0).unsqueeze(0).expand(3, -1, -1, -1).contiguous()

        self.register_buffer('window', window)

    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, predictions, ground_truth):
        l1_loss = self.l1(predictions, ground_truth)
        ssim_loss = 1 - self.ssim(predictions, ground_truth)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
    
    @torch.compile
    def ssim(self, x_batch, y_batch):
        '''
        Calculate and return the average SSIM between img1 and img2
        '''
        p = self.window_size // 2
        x_batch = F.pad(x_batch, (p, p, p, p), mode='reflect')
        y_batch = F.pad(y_batch, (p, p, p, p), mode='reflect')

        mu_x = F.conv2d(x_batch, self.window, groups=3)
        mu_y = F.conv2d(y_batch, self.window, groups=3)

        var_x = F.relu(F.conv2d(x_batch.pow(2), self.window, groups=3) - mu_x.pow(2))
        var_y = F.relu(F.conv2d(y_batch.pow(2), self.window, groups=3) - mu_y.pow(2))

        sigma_xy = F.conv2d(x_batch * y_batch, self.window, groups=3) - (mu_x * mu_y)

        ssim = (2 * mu_x * mu_y + 0.0001) * (2 * sigma_xy + 0.0009) / ((mu_x.pow(2) + mu_y.pow(2) + 0.0001) * (var_x + var_y + 0.0009))
                
        return torch.mean(ssim)