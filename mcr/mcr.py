import warnings

import numpy as np
from scipy.optimize import nnls

from .regressors import RegressorIterator, nnls_fraction


def tot_rmse(diff):
    return np.sqrt(np.square(diff).mean(axis=0)).sum()


class MCR:
    """
    Multivariate Curve Resolution by Alternating Least Squares

    M' = C·B'

    """

    def __init__(
        self,
        cregr=nnls_fraction,
        bregr=RegressorIterator(lambda x, y: nnls(x, y)[0]),
        tol=1e-6,
        chkpnt=10,
        maxiter=50,
        errfunc=tot_rmse,
    ):

        self.maxiter = maxiter
        self.errfunc = errfunc

        self.cregr = cregr
        self.bregr = bregr

        self.tol = tol
        self.chkpnt = chkpnt
        self.converged = False
        self.bmode = False

    def fit(self, M, C=None, B=None, debug=True):

        self.M = M.copy()

        if C is None:
            if B is None:
                raise Exception("You should provide either C or B")
            else:
                self.bmode = True
                self.B = B.copy()
                self.C = self.cregr.fit_transform(self.B, self.M)
        else:
            if B is not None:
                warnings.warn("You provided both B & C, discarding B")
            self.C = C.copy()

        if debug:
            self.error = np.zeros(self.maxiter // self.chkpnt)

        for i in range(self.maxiter):

            self.B = self.bregr.fit_transform(self.C, self.M.T)
            self.C = self.cregr.fit_transform(self.B, self.M)

            if i % self.chkpnt == 0:
                error = self.errfunc(self.B @ self.C.T - self.M)

                if debug:
                    self.error[i // self.chkpnt] = error

                if error < self.tol:
                    self.converged = True
                    self.error = self.error[:i//self.chkpnt+1]
                    break

        return self

    def fit_transform(self, *args):
        self.fit(*args)
        return self.C if self.bmode else self.B
