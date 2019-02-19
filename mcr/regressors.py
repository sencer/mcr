import numpy as np
from scipy.optimize import nnls


class RegressorIterator:
    def __init__(self, regressor, *args):
        self.regressor = regressor
        self.args = args

    @property
    def coef_(self):
        if self.X_ is None:
            return None
        else:
            return self.X_.T

    def fit(self, A, B):

        N = B.shape[-1]

        if len(self.args) == 0:
            self.X_ = np.r_[[self.regressor(A, B[:, col]) for col in range(N)]].T
        else:
            self.X_ = np.r_[
                [
                    self.regressor(A, B[:, col], [arg[col] for arg in self.args])
                    for col in range(N)
                ]
            ].T

        return self

        self.X_ = np.zeros((A.shape[-1], N))
        self.residual_ = np.zeros(N)

        for col in range(N):
            self.X_[:, col], self.residual_[col] = self.regressor(A, B[:, col])

        return self

    def fit_transform(self, A, B):
        self.fit(A, B)
        return self.coef_



@RegressorIterator
def nnls_fraction(A, b):
    return nnls(np.r_[A, np.ones((1, A.shape[1]))], np.r_[b, 1])[0]
