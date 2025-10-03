import numpy as np


def get_near_psd(dt, eps):
    # get the matrix dimension
    k = dt.shape[0]

    # extract the variance matrix
    var = np.diag(np.diag(dt))
    # transform the covariance into correlation matrix
    dt = np.linalg.inv(np.sqrt(var)) @ dt @ np.linalg.inv(np.sqrt(var))

    eigenval, eigenvec = np.linalg.eig(dt)
    eigenval_p = np.maximum(eigenval, eps)

    T = np.zeros((k, k))

    for i in range(0, k):
        t_i = 0
        for j in range(0, k):
            t_i = t_i + eigenvec[i, j] * eigenvec[i, j] * eigenval_p[j]
        T[i, i] = 1 / t_i

    lam = np.diag(eigenval_p)
    B = np.sqrt(T) @ eigenvec @ np.sqrt(lam)

    C = B @ B.T

    return np.sqrt(var) @ C @ np.sqrt(var)
