import torch
import torch.nn as nn
import numpy as np

activation = nn.LeakyReLU


def swap_halves(x):
    a, b = x.split(x.shape[0]//2)
    return torch.cat([b, a])


def lerp(start, end, weights):
    # torch.lerp only support scalar weight
    return start + weights * (end - start)


def l2_norm(x):
    return torch.mean(x**2)


# authors use this initializer, but it doesn't seem essential
def Initializer(model, slope=0.2):
    for m in model:
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            std = 1 / np.sqrt((1 + slope ** 2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)
            m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self, scales, depth, latent, colors):
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.Conv2d(colors, depth, 1, padding=0))  # padding = 1 stupid fuck
        kp = depth
        for scale in range(scales):
            k = depth << scale
            layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
            layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
            layers.append(nn.AvgPool2d(2))
            kp = k
        k = depth << scales
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.append(nn.Conv2d(k, latent, 3, padding=1))
        self.encoder = nn.Sequential(*layers)
        Initializer(self.encoder)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, scales, depth, latent, colors):
        super(Decoder, self).__init__()
        layers = []
        kp = latent
        for scale in range(scales - 1, -1, -1):
            k = depth << scale
            layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
            layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
            layers.append(nn.Upsample(scale_factor=2))
            kp = k
        layers.extend([nn.Conv2d(kp, depth, 3, padding=1), activation()])
        layers.append(nn.Conv2d(depth, colors, 3, padding=1))
        self.decoder = nn.Sequential(*layers)
        Initializer(self.decoder)

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, scales, depth, latent, colors):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(scales, depth, latent, colors)
        self.decoder = Decoder(scales, depth, latent, colors)

    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x, y


class Discriminator(nn.Module):
    def __init__(self, scales, depth, latent, colors):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(scales, depth, latent, colors)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.mean(x, -1)
        return x


class Padding(object):

    def __call__(self, x):
        # print(np.array(x).shape)
        x = np.pad(np.array(x), ((2, 2), (2, 2)), mode='constant')
        # print(x.shape)
        return x


class Unsqueeze(object):

    def __call__(self, x):
        # print(x.shape)
        x = np.expand_dims(x, axis=2)
        # print(x.shape)
        return x