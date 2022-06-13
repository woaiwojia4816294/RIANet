import numpy as np
import math


def psnr(img, groundTruth):
    mse = np.mean((img.astype(np.float64) - groundTruth.astype(np.float64)) ** 2)  # dui
    if mse == 0:
        return np.inf
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
