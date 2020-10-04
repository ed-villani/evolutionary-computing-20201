import numpy as np


def rastrigin(x):
    Q = np.identity(len(x))
    X = Q @ x
    n = len(x)
    F = 0

    for i in range(n):
        F = F + X[i] ** 2 - 10 * np.cos(2 * np.pi * X[i])
    return F

