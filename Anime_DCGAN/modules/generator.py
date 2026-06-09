"""Generator module for the Anime DCGAN, producing images from random noise via transposed convolutions."""
import torch
import torch.nn as nn
from torchvision.utils import save_image


class generator(nn.Module):
    """DCGAN generator that upsamples a noise vector into an image using stacked transposed convolution blocks."""

    def __init__(self, input_dim, hidden_dim, im_dim):
        """Initialize the generator with a sequential stack of transposed convolution blocks.

        input_dim is the noise latent size, hidden_dim is the base channel multiplier, im_dim is output channels.
        """
        super(generator, self).__init__()
        self.generator = nn.Sequential(
            self.generator_block(input_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0, last_layer=False),
            self.generator_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1, last_layer=False),
            self.generator_block(hidden_dim * 4, hidden_dim*2, kernel_size=4, stride=2, padding=1, last_layer=False),
            self.generator_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, last_layer=False),
            self.generator_block(hidden_dim, im_dim, kernel_size=4, stride=2, padding=1, last_layer=True),
        )

    @staticmethod
    def generator_block(in_channels, out_channels, kernel_size, stride, padding, last_layer=False):
        """Build a single transposed convolution block.

        Uses BatchNorm and LeakyReLU for intermediate layers, and Tanh activation for the final output layer.
        """
        if last_layer == False:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))

        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Tanh())

        return block

    def forward(self, noise):
        """Reshape noise to a spatial tensor and pass it through the generator to produce a fake image."""
        noise = noise.view(noise.shape[0], noise.shape[1], 1, 1)

        xd = self.generator(noise)

        return xd


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# noise = torch.normal(0, 1, size=(2, 64)).to(device)
#
# noise = noise.view(noise.shape[0], noise.shape[1], 1, 1)
#
# gen = generator(64, 128, 1).to(device)
#
# image = gen(noise)
#
# print(image.shape)

# gen_image=gen(noise)
# epoch=10777777

# save_image(gen_image.view(gen_image.size(0), 1, 28, 28), '../gen_images/sample_' + str(epoch) + '.png')

# print(gen_image.shape)

def gen_loss(gen, disc, criterion, real_im, noise_dim, device):
    """Compute generator loss by fooling the discriminator: generates fake images and measures how real they appear."""
    noise_vec = torch.randn(len(real_im), noise_dim, device=device)

    fake_images = gen(noise_vec)

    pred_fakes = disc(fake_images)

    ground_real = torch.ones_like(pred_fakes)

    loss = criterion(pred_fakes, ground_real)

    return loss