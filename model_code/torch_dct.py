import numpy as np
import torch
import torch.nn as nn


def dct_2d(x, norm=None):
    return torch.fft.fft2(x)


def idct_2d(X, norm=None):
    return torch.fft.ifft2(X)
