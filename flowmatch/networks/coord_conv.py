import numpy as np
import torch
import torch.nn as nn

def concatenate_coords(inp):
    """Computes and concatenates x and y coordinates to input.

    x and y coordinates range from [-0.5, 0.5].
    """
    batch_size, in_channels, height, width = inp.shape

    coords = np.indices((height, width), dtype=np.float32) + 0.5  # shape is (2, height, width)
    coords = np.flip(coords, 0).copy()  # so that coordinates are in (x, y) format
    coords[0, :, :] /= width  # normalize to [0, 1]
    coords[1, :, :] /= height  # normalize to [0, 1]
    coords -= 0.5  # normalize to [-0.5, 0.5]

    coords = torch.from_numpy(coords).type(inp.type())  # convert to torch tensor
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
    coords = coords.type(inp.dtype)

    ret = torch.cat((inp, coords), dim=1)  # concatenate inp and coords along feature dim
    return ret


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super(CoordConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, *args, **kwargs)  # two more in_channels

    def forward(self, x):
        x = concatenate_coords(x)
        x = self.conv(x)
        return x


class CoordConvTranspose2d(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super(CoordConvTranspose2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels + 2, *args, **kwargs)  # two more in_channels

    def forward(self, x):
        x = concatenate_coords(x)
        x = self.conv_transpose(x)
        return x
