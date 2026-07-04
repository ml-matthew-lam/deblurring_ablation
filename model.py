import torch
import torch.nn as nn
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        shortened_length = max(n_channels//4, 16)
        self.conv1 = nn.Conv2d(n_channels, shortened_length, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(shortened_length, n_channels, kernel_size=1, bias=False)
    def forward(self, x):
        output = self.pool(x)
        output = self.conv1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output = torch.sigmoid(output)
        output = x * output
        return output

class SCA(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False)
    def forward(self, x):
        output = self.pool(x)
        output = self.conv(output)
        output = x * output
        return output


class BaselineBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size = 1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, groups = n_channels, kernel_size = 3, padding = 1, padding_mode='reflect')
        self.attn = CA(n_channels)
        self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=1)
    def forward(self, x):
        residual = x

        # LayerNorm
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.attn(x)
        x = self.conv3(x)
        return residual + x


class NAFBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.conv1 = nn.Conv2d(n_channels, 2*n_channels, kernel_size = 1)
        self.conv2 = nn.Conv2d(2*n_channels, 2*n_channels, groups = 2*n_channels, kernel_size = 3, padding = 1, padding_mode='reflect')
        self.attn = SCA(n_channels)
        self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=1)
    def forward(self, x):
        residual = x

        # LayerNorm
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)

        x = self.conv1(x)
        x = self.conv2(x)

        # SimpleGate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2

        x = self.attn(x)
        x = self.conv3(x)
        return residual + x
    

class UNet(nn.Module): # note: height and width of image both need to be divisible by 8
    def __init__(self, block, init_channels):
        super().__init__()

        # encoder
        self.right1 = nn.Sequential(nn.Conv2d(3, init_channels, kernel_size=3, padding=1, padding_mode='reflect'), block(init_channels))
        self.down1 = nn.Conv2d(init_channels, 2*init_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.right2 = block(2*init_channels)
        self.down2 = nn.Conv2d(2*init_channels, 4*init_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.right3 = block(4*init_channels)
        self.down3 = nn.Conv2d(4*init_channels, 8*init_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')

        # bottleneck
        self.right4 = block(8*init_channels)
        self.right5 = block(8*init_channels)
        self.right6 = block(8*init_channels)
        self.right7 = block(8*init_channels)


        # decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8*init_channels, 4*init_channels, kernel_size=1)
        )
        self.right8 = nn.Sequential(
            nn.Conv2d(8*init_channels, 4*init_channels, kernel_size=1),
            block(4*init_channels)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(4*init_channels, 2*init_channels, kernel_size=1)
        )
        self.right9 = nn.Sequential(
            nn.Conv2d(4*init_channels, 2*init_channels, kernel_size=1),
            block(2*init_channels)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(2*init_channels, init_channels, kernel_size=1)
        )
        self.right10 = nn.Sequential(
            nn.Conv2d(2*init_channels, init_channels, kernel_size=1),
            block(init_channels), 
            nn.Conv2d(init_channels, 3, kernel_size=1)
        )

    def forward(self, x):
        residual = x

        # encoder

        x = self.right1(x)
        copy1 = x

        x = self.down1(x)
        x = self.right2(x)
        copy2 = x

        x = self.down2(x)
        x = self.right3(x)
        copy3 = x

        x = self.down3(x)

        # bottleneck
        x = self.right4(x)
        x = self.right5(x)
        x = self.right6(x)
        x = self.right7(x)
        
        # decoder

        x = self.up1(x)
        x = torch.cat((copy3, x), dim=1)

        x = self.right8(x)
        x = self.up2(x)
        x = torch.cat((copy2, x), dim=1)

        x = self.right9(x)
        x = self.up3(x)
        x = torch.cat((copy1, x), dim=1)

        x = self.right10(x)

        return residual + x



# if __name__ == "__main__":
#     blocks = [BaselineBlock, NAFBlock]
#     for block in blocks:
#         model = UNet(block, 32)
#         output = model(torch.randn(8, 3, 256, 256))
#         print(output.size())

