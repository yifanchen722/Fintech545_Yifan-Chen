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


def Frobeniusnorm(A):
    return np.sum(np.square(A))


def Frobeniusnorm_W(A, W):
    return np.sum(np.square(np.linalg.sqrtm(W) @ A @ np.linalg.sqrtm(W)))


def PUA(A):
    I = np.eye(A.shape[0])
    theta = np.diag(A - I)
    return A - np.diag(theta)


def PUA_W(A, W):
    I = np.eye(A.shape[0])
    theta = np.linalg.inv(np.square(np.linalg.inv(W))) @ np.diag(A - I)
    return A - np.linalg.inv(W) @ np.diag(theta) @ np.linalg.inv(W)


def APLUS(A):
    eigenval, eigenvec = np.linalg.eig(A)
    eigenval_p = np.maximum(eigenval, 1e-8)
    lam = np.diag(eigenval_p)
    return eigenvec @ lam @ eigenvec.T


def PSA(A):
    return APLUS(A)


def PSA_W(A, W):
    return (
        np.linalg.sqrtm(np.linalg.inv(W))
        @ APLUS(np.linalg.sqrtm(W) @ A @ np.linalg.sqrtm(W))
        @ np.linalg.sqrtm(np.linalg.inv(W))
    )
