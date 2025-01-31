import numpy as np

class LinRegSolver:
    def __init__(self, x: np.ndarray, y: np.ndarray, add_intercept_variable: bool = True) -> None:
        assert 2 <= x.ndim <= 3
        assert 0 <= x.ndim - y.ndim <= 1

        if add_intercept_variable:
            x = np.concatenate([np.ones((*x.shape[:-1], 1)), x], axis = -1)

        self.x = x
        self.y = y
        self.multivariate = x.ndim == y.ndim


    def ols(self) -> np.ndarray:
        print(self.x.shape)
        print(self.y.shape)
        return np.linalg.lstsq(self.x, self.y)[0]


    
