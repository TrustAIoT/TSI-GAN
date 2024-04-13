import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from torchvision import datasets, transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    def __init__(self, im_chan=2, z_dim=100, hidden_dim=48):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(im_chan, hidden_dim, kernel_size=7,
                      stride=3, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, z_dim, kernel_size=2, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(hidden_dim*8),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(hidden_dim*8, z_dim, kernel_size=2, stride=1, padding=0, bias=False),
        )

    def forward(self, image):
        return self.enc(image)


class Generator(nn.Module):
    def __init__(self, z_dim=100, im_chan=2, hidden_dim=24):
        super(Generator, self).__init__()
        # Z shape = batch_size x z_dim x 1 x 1
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_dim * 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size = (hidden_dim*4 x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # H = (4-1)*2 - 2 + (4-1) + 1 = 8
            # state size = (hidden_dim*2) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=5, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # H = (8-1)*1 - 2*0 + (4-1) + 1 = 7 -2 + 4 = 11
            # state size = (hidden_dim) x 11 x 11

            nn.ConvTranspose2d(hidden_dim, im_chan, 7, 3, 0, bias=False),
            nn.Tanh(),

            # nn.ConvTranspose2d(hidden_dim, im_chan, 4, 2 , 0, bias=False),
            # nn.Tanh(),
            # H = (11-1)*2 - 2*0 + (4-1) + 1 = 24
            # state size = 2 x 24 x 24
        )

    def forward(self, z):
        # x = noise.view(len(noise), self.im_chan, self.z_dim, self.z_dim)
        return self.dec(z)


class Critic_Z(nn.Module):
    def __init__(self, z_dim=100):
        super(Critic_Z, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.LayerNorm((z_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim // 2, z_dim // 4),
            nn.LayerNorm((z_dim // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim // 4, 1),
        )

    def forward(self, z):
        z = z.squeeze()
        return self.disc(z)


# noinspection PyPep8Naming
class Critic_X(nn.Module):
    def __init__(self, cfg, im_chan=2, hidden_dim=48):
        super(Critic_X, self).__init__()
        # X size: 2 x 24 x 24 (2 channels)
        # here we use gradient penalty for WGAN, so according to the paper
        # we should use LayerNorm instead of BatchNorm
        self.in_dim = 64
        self.cfg = cfg
        self.disc = nn.Sequential(
            nn.Conv2d(im_chan, hidden_dim, kernel_size=7,
                      stride=3, padding=0, bias=False),
            nn.LayerNorm((hidden_dim, *([self.get_out_dim(0)] * 2))),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=3, padding=0, bias=False),
            nn.LayerNorm((hidden_dim * 2, *([self.get_out_dim(1)] * 2))),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LayerNorm((hidden_dim * 4, *([self.get_out_dim(2)] * 2))),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, 1, kernel_size=2, stride=1, padding=0, bias=False),
            # nn.LayerNorm((hidden_dim*8, *([self.get_out_dim(3)]*2))),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(hidden_dim*8, 1, kernel_size=2, stride=1, padding=0, bias=False),
        )

    def get_out_dim(self, i):
        H_out = (self.in_dim + 2 * self.cfg['conv_padding'][i] -
                 (self.cfg['conv_kernel_size'][i] - 1) - 1) // self.cfg['conv_stride'][i] + 1
        self.in_dim = H_out
        # print(H_out)
        return H_out

    def forward(self, image):
        return self.disc(image)
