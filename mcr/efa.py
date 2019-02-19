import numpy as np
from scipy.linalg import svdvals

def efa(data, ncomp=2, plot=False):

    data = data.copy().T
    nrows = len(data)

    ef = np.zeros((nrows, nrows))
    eb = np.zeros((nrows, nrows))
    for i in range(1, nrows + 1):
        ef[i - 1, :i] = np.square(svdvals(data[:i]))
        eb[i - 1, i - 1 :] = np.square(svdvals(data[i - 1 :])[::-1])

    C = np.stack([ef[:, :ncomp], eb[:, -ncomp:]]).min(axis=0)

    if plot:
        import matplotlib.pyplot as plt
        X = np.arange(nrows)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        for row in np.log10(ef.T):
            ax1.plot(row, "k")
        for row in np.log10(eb.T):
            ax1.plot(row, "r")

        for row in C.T:
            ax2.plot(row)

        fig.tight_layout()

    return C
