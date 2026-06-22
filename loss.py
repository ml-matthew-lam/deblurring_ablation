import torch
import torch.nn as nn
import torch.nn.functional as F

class LossCombination(nn.Module)
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha # alpha controls the proportion of L1 vs SSIM in the total loss combination
        self.l1 = nn.L1Loss()
    
    def forward(self, predictions, ground_truth):
        l1_loss = self.l1(predictions, ground_truth)
        ssim_loss = 1 - self.ssim(predictions, ground_truth)
        return self.alpha*l1_loss + (1-self.alpha)*ssim_loss
    
    def ssim(self, img1, img2):
        # not finished: calculate SSIM between img1 and img2