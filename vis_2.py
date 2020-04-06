# import torch
# from multiprocessing import cpu_count
# from torchvision.datasets import MNIST
# from torchvision.transforms import Compose, ToTensor, Normalize
# from torch.utils.data import DataLoader
# from utils import Padding, Unsqueeze
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as mpimg

# Just run it once!
# batch_size = 256
# use_cuda = torch.cuda.is_available()
# kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if use_cuda else {}
# device = torch.device("cuda" if use_cuda else "cpu")
# ds = MNIST('data', train=True, download=False,
#            transform=Compose([Padding(),  Unsqueeze(), ToTensor(), Normalize((0.1307,), (0.3081,))]))
# ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, **kwargs)
#
# for i, item in enumerate(ds_loader):
#     image, label = item[0], item[1]
#     print(image.shape)
#     for idx in range(0, 10):
#         first_idx = (item[1] == idx).nonzero()[0].squeeze()
#         plt.imsave(str(idx)+'.png', image[first_idx].squeeze().data.numpy())
#     break

# x1 = Image.open(str(0)+'.png')
x1 = mpimg.imread(str(0)+'.png')
# plt.imshow(x1)
# plt.show()

# x1 = np.array(x1[:, :, 0])
# print(x1.shape)
# x1 = np.expand_dims(x1, axis=0)
# print(x1.shape)
# x1 = np.repeat(x1, 10, axis = 0)
# print(x1.shape)
