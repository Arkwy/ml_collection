import numpy as np
from matplotlib import pyplot as plt

from solvers import LinRegSolver

N = 100
NOISE = 0

rng = np.random.default_rng()

x = rng.random(N)

y = 2 * x + 3

noise = rng.standard_normal(x.shape) * NOISE
x += noise
X = np.stack([np.ones(N), x], axis=1)

A = X.T @ X
Am1 = 1 / (A[0,0]*A[1,1] - A[0,1]*A[1,0]) * np.array([[A[1,1], -A[0,1]], [-A[1,0], A[0,0]]])

s = Am1 @ X.T @ y 

print(LinRegSolver(x[:, None], y).ols())
