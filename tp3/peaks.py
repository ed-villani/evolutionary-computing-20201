import numpy as np


def peaks(x, y):
    exp_1 = 3 * (1 - x) ** 2 * np.exp(- (x ** 2) - (y + 1) ** 2)
    exp_2 = 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-(x ** 2) - (y ** 2))
    exp_3 = (1 / 3) * np.exp(-(x + 1) ** 2 - (y ** 2))
    return exp_1 - exp_2 - exp_3
