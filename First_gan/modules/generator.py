"""Generator module for a simple fully-connected GAN on image data."""
import torch
import torch.nn as nn
from torchvision.utils import save_image


class generator(nn.Module):
    """Fully-connected generator that maps a noise vector to a flat image."""

    def __init__(self, input_dim, hidden_dim, im_dim):
        """Build the generator network given input, hidden, and output dimensions."""
        super(generator, self).__init__()
        self.generator = nn.Sequential(
            self.generator_block(input_dim, hidden_dim),
            self.generator_block(hidden_dim, hidden_dim*2),
            self.generator_block(hidden_dim*2, hidden_dim*4),
            self.generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, im_dim),
            nn.Sigmoid()
        )

    def generator_block(self, input, output):
        """Return a Linear -> BatchNorm1d -> ReLU block."""
        block = nn.Sequential(
            nn.Linear(input, output),
            nn.BatchNorm1d(output),
            nn.ReLU(inplace=True))
        return block

    def forward(self, noise):
        """Pass a noise vector through the generator and return the generated image."""
        xd = self.generator(noise)
        return xd


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# gen = generator(64,128,784).to(device)
# noise= torch.normal(0,1, size=(2, 64)).to(device)

# gen_image=gen(noise)
# epoch=10

# save_image(gen_image.view(gen_image.size(0), 1, 28, 28), '../gen_images/sample_' + str(epoch) + '.png')

# print(gen_image.shape)


def gen_loss(gen, disc, criterion, real_im, noise_dim, device):
    """Compute the generator loss by fooling the discriminator with generated images."""
    noise_vec = torch.normal(0, 1, size=(len(real_im), noise_dim))
    noise_vec = noise_vec.to(device)

    fake_images = gen(noise_vec)

    pred_fakes = disc(fake_images)

    ground_real = torch.ones_like(pred_fakes)

    gen_loss = criterion(pred_fakes, ground_real)

    return gen_loss