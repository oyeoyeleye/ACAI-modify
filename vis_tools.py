from mosaic import make_mosaic
import numpy as np
import torch


def reconstruct(x, encoder, decoder):
    out = decoder(encoder(x))
    return make_mosaic(out.cpu().data.numpy().squeeze())


def interpolate_2(x, encoder, decoder, side=8, device='cpu'):
    z = encoder(x)
    z = z.data.cpu().numpy()

    a, b = z[:side], z[-side:]
    z_interp = [a * (1-t) + b * t for t in np.linspace(0, 1, side-2)]
    z_interp = np.vstack(z_interp)
    x_interp = decoder(torch.FloatTensor(z_interp).to(device))
    x_interp = x_interp.cpu().data.numpy()

    x_fixed = x.data.cpu().numpy()
    all = []
    all.extend(x_fixed[:side])
    all.extend(x_interp)
    all.extend(x_fixed[-side:])

    return make_mosaic(np.asarray(all).squeeze())


def interpolate_4(x, encoder, decoder, side=8, device='cpu'):
    z = encoder(x)
    z = z.data.cpu().numpy()

    n = side*side
    xv, yv = np.meshgrid(np.linspace(0, 1, side),
                         np.linspace(0, 1, side))
    xv = xv.reshape(n, 1, 1, 1)
    yv = yv.reshape(n, 1, 1, 1)

    z_interp = z[0]*(1-xv)*(1-yv) + z[1]*xv*(1-yv) + z[2]*(1-xv)*yv + z[3]*xv*yv

    x_fixed = x.data.cpu().numpy()
    x_interp = decoder(torch.FloatTensor(z_interp).to(device))
    x_interp = x_interp.data.cpu().numpy()
    x_interp[0] = x_fixed[0]
    x_interp[side-1] = x_fixed[1]
    x_interp[n-side] = x_fixed[2]
    x_interp[n-1] = x_fixed[3]

    return make_mosaic(x_interp.squeeze())


# random samples based on a reference distribution
def random_samples(sample, encoder, decoder, device='cpu'):
    z = encoder(sample)
    z = z.data.cpu().numpy()
    z_sample = np.random.normal(loc=z.mean(axis=0), scale=z.std(axis=0), size=z.shape)
    x_sample = decoder(torch.FloatTensor(z_sample).to(device))
    x_sample = x_sample.data.cpu().numpy()
    return make_mosaic(x_sample.squeeze())


def status(x, encoder, decoder):
    chunks = [
        reconstruct(x, encoder, decoder),
        interpolate_2(x, encoder, decoder),
        interpolate_4(x, encoder, decoder), random_samples(x, encoder, decoder)]
    chunks = [np.pad(e, (0, 1), mode='constant', constant_values=255) for e in chunks]
    return make_mosaic(chunks)
