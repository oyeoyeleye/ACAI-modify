import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torchvision.transforms import Compose, ToTensor, Normalize
# x = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
# y = np.swapaxes(x, 0, 2)
# print(x.shape)
# print(y.shape)
#
# print(x)
# print('\n')
# print(y)

# vec = torch.FloatTensor([-3.2, 1000.0, 10.0, 639.0])
# lower_bound = 0.0
# upper_bound = 640.0
# lower_bound_vec = torch.ones_like(vec) * lower_bound
# upper_bound_vec = torch.ones_like(vec) * upper_bound
# zeros_vec       = torch.zeros_like(vec)
#
# def where(cond, x_1, x_2):
#     cond = cond.float()
#     return (cond * x_1) + ((1-cond) * x_2)
#
# print(vec, '\n')
# vec = where(vec < lower_bound_vec, zeros_vec, vec)
# print(vec, '\n')
# vec = where(vec > upper_bound_vec, zeros_vec, vec)
# print(vec, '\n')
#
# in_bound_indices = torch.nonzero(vec).squeeze(1)
# print(in_bound_indices, '\n')
# vec = torch.index_select(vec, 0, in_bound_indices)
# print(vec, '\n')
#
# use_cuda = torch.cuda.is_available()
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# device = torch.device("cuda" if use_cuda else "cpu")
# ds = MNIST('data', train=True, download=False,
#            transform=Compose([ToTensor()]))
# ds_loader = DataLoader(ds, batch_size=1024, shuffle=True, **kwargs)
#
# for i, sample in enumerate(ds_loader):
#     x, y = sample[0], sample[1].float()
#     # print((y==2).nonzero())
#     print(x[(y==2).nonzero()])

alpha = np.arange(0, .5, .01)
print(alpha)
print(np.concatenate([alpha, alpha], axis=0))