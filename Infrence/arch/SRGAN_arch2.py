"""SRGAN architecture definitions used during inference.

Contains the Generator and Discriminator networks along with
supporting residual and upsample blocks, following the original
SRGAN paper design.
"""
import torch
from torch import nn
from torchinfo import summary


class Residual_block(nn.Module):
    """A single residual block with two conv layers, batch norm, and PReLU activation."""

    def __init__(self, channels):
        """Initialize residual block with the given number of channels."""
        super(Residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Apply residual block and add the skip connection."""
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.act(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res


class Upsample_block(nn.Module):
    """Upsampling block using pixel shuffle for spatial resolution increase."""

    def __init__(self, channels, upscale):
        """Initialize upsample block with the given channel count and upscale factor."""
        super(Upsample_block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels * upscale ** 2, kernel_size=3, stride=1, padding=1)
        self.pshuff = nn.PixelShuffle(upscale)
        self.act = nn.PReLU(num_parameters=channels)

    def forward(self, x):
        """Apply conv, pixel shuffle, and PReLU to upsample the input."""
        up = self.conv1(x)
        up = self.pshuff(up)
        up = self.act(up)
        return up


class Generator(nn.Module):
    """SRGAN Generator that upscales a low-resolution image by 4x using residual blocks and pixel shuffle."""

    def __init__(self):
        """Build the generator with 16 residual blocks and two upsample stages."""
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(num_parameters=64)
        )
        self.block2 = nn.Sequential(*[Residual_block(64) for _ in range(16)])
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block4 = nn.Sequential(Upsample_block(64, 2), Upsample_block(64, 2))
        self.block5 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        """Run the forward pass and return the super-resolved image scaled to [-1, 1]."""
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2) + block1
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        return torch.tanh(block5)


class Discriminator(nn.Module):
    """SRGAN Discriminator that classifies images as real or generated."""

    def __init__(self):
        """Build the discriminator with a series of strided conv layers and a linear classifier."""
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),

        )

    def forward(self, x):
        """Return raw discriminator logits for the input image batch."""
        return self.disc(x)


# ############################
# x = torch.randn((5, 3, 24, 24))
# gen = Generator()
# summary(gen, input_size=(16, 3, 64, 64))
# gen_out = gen(x)
# print(gen_out.size())
# disc = Discriminator()
# disc_out = disc(gen_out)
# print(disc_out.size())
# ##############################