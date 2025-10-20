import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t


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


def get_ew_cov(data_path, lam, has_date, is_rate):
    dt1 = pd.read_csv(data_path)

    if is_rate & has_date:
        # 如果已经是收益率数据，则不需要计算收益率，直接使用dt1
        # there is a date column, use iloc[:,1:] instead of iloc[:, :] to exclude the date column
        dt2 = dt1.copy().iloc[:, 1:]
        dt2.index = range(dt2.shape[0])
    elif is_rate & (not has_date):
        # 如果已经是收益率数据，则不需要计算收益率，直接使用dt1
        # there is no date column, use iloc[:, :] to include the date column
        dt2 = dt1.copy().iloc[:, :]
        dt2.index = range(dt2.shape[0])
    elif (not is_rate) & has_date:
        # 需要计算收益率
        # there is a date column, use iloc[1:,1:] instead of iloc[1, :] to exclude the date column
        dt2 = dt1.copy().iloc[1:, 1:]
        dt2.index = range(dt2.shape[0])
        dt2 = (dt2 - dt1.iloc[:-1, 1:]) / dt1.iloc[:-1, 1:]
    else:
        # 需要计算收益率
        # there is no date column, use iloc[1:, :] to include the date column
        dt2 = dt1.copy().iloc[1:, :]
        dt2.index = range(dt2.shape[0])
        dt2 = (dt2 - dt1.iloc[:-1, :]) / dt1.iloc[:-1, :]

    # weight vector
    wts = (1 - lam) * lam ** np.arange(dt2.shape[0] - 1, -1, -1)
    # normalize weights
    wts = wts / sum(wts)

    # covariance matrix
    mean_vec = []
    for col in range(0, dt2.shape[1]):
        mean_vec.append(sum(dt2.iloc[:, col] * wts))

    # placeholder for covariance matrix
    cov = np.zeros((dt2.shape[1], dt2.shape[1]))
    for i in range(0, dt2.shape[1]):
        for j in range(0, dt2.shape[1]):
            cov[i, j] = ((dt2.iloc[:, i] - mean_vec[i]) * wts).T @ (
                dt2.iloc[:, j] - mean_vec[j]
            )
    return cov


def get_ew_corr(data_path, lam, has_date, is_rate):
    dt1 = pd.read_csv(data_path)

    if is_rate & has_date:
        # 如果已经是收益率数据，则不需要计算收益率，直接使用dt1
        # there is a date column, use iloc[:,1:] instead of iloc[:, :] to exclude the date column
        dt2 = dt1.copy().iloc[:, 1:]
        dt2.index = range(dt2.shape[0])
    elif is_rate & (not has_date):
        # 如果已经是收益率数据，则不需要计算收益率，直接使用dt1
        # there is no date column, use iloc[:, :] to include the date column
        dt2 = dt1.copy().iloc[:, :]
        dt2.index = range(dt2.shape[0])
    elif (not is_rate) & has_date:
        # 需要计算收益率
        # there is a date column, use iloc[1:,1:] instead of iloc[1, :] to exclude the date column
        dt2 = dt1.copy().iloc[1:, 1:]
        dt2.index = range(dt2.shape[0])
        dt2 = (dt2 - dt1.iloc[:-1, 1:]) / dt1.iloc[:-1, 1:]
    else:
        # 需要计算收益率
        # there is no date column, use iloc[1:, :] to include the date column
        dt2 = dt1.copy().iloc[1:, :]
        dt2.index = range(dt2.shape[0])
        dt2 = (dt2 - dt1.iloc[:-1, :]) / dt1.iloc[:-1, :]

    # weight vector
    wts = (1 - lam) * lam ** np.arange(dt2.shape[0] - 1, -1, -1)
    # normalize weights
    wts = wts / sum(wts)

    # covariance matrix
    mean_vec = []
    for col in range(0, dt2.shape[1]):
        mean_vec.append(sum(dt2.iloc[:, col] * wts))

    # placeholder for covariance matrix
    cov = np.zeros((dt2.shape[1], dt2.shape[1]))
    for i in range(0, dt2.shape[1]):
        for j in range(0, dt2.shape[1]):
            cov[i, j] = ((dt2.iloc[:, i] - mean_vec[i]) * wts).T @ (
                dt2.iloc[:, j] - mean_vec[j]
            )

    # correlation matrix
    corr = np.zeros((dt2.shape[1], dt2.shape[1]))
    for i in range(0, dt2.shape[1]):
        for j in range(0, dt2.shape[1]):
            corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    return corr


def higham(data_path, maxiter, tol):
    C = np.loadtxt(data_path, delimiter=",", skiprows=1)
    var = np.diag(np.diag(C))
    C = np.linalg.inv(np.sqrt(var)) @ C @ np.linalg.inv(np.sqrt(var))

    deltaS = np.zeros((C.shape[0], C.shape[1]))
    Y = C
    gamma_1 = np.finfo(np.float64).max
    gamma_2 = np.finfo(np.float64).max
    R = np.zeros((C.shape[0], C.shape[1]))
    X = np.zeros((C.shape[0], C.shape[1]))

    for k in range(1, maxiter + 1):
        gamma_1 = gamma_2
        R = Y - deltaS
        X = PSA(R)
        deltaS = X - R
        Y = PUA(X)
        gamma_2 = Frobeniusnorm(Y - C)
        print("iter:", k, "gamma_2:", gamma_2)
        if abs(gamma_1 - gamma_2) <= tol:
            break

    Y = np.sqrt(var) @ Y @ np.sqrt(var)
    return Y


def tell_psd(data_path):
    df = pd.read_csv(data_path, sep=",")
    data = df.values.astype(float)
    print(data)

    C_hat = get_near_psd(data, eps=1e-8)
    print(C_hat)
    if np.allclose(data, C_hat, rtol=1e-6, atol=1e-9):
        return True
    else:
        return False


def summ_stats(data):
    print("Mean:", data.mean())
    print("Variance:", data.var(ddof=1))
    print("Standard Deviation:", data.std(ddof=1))
    print("Skewness:", data.skew())
    print("Kurtosis:", data.kurt())


def fit_t_distribution(x):

    nu, mu, sigma = t.fit(x)

    print("mu\tsigma\tnu")
    print(f"{mu}\t{sigma}\t{nu}")

    mean_t = mu if nu > 1 else np.nan
    var_t = sigma**2 * nu / (nu - 2) if nu > 2 else np.nan
    skew_t = 0 if nu > 3 else np.nan
    kurt_t = 6 / (nu - 4) if nu > 4 else np.nan  # excess kurtosis

    print("\nTheoretical moments:")
    print(f"Mean: {mean_t}")
    print(f"Variance: {var_t}")
    print(f"Skewness: {skew_t}")
    print(f"Kurtosis: {kurt_t}")


def fit_normal_distribution(x):
    mu, sigma = norm.fit(x)

    print("mu\tsigma")
    print(f"{mu}\t{sigma}")

    mean_n = mu
    var_n = sigma**2
    skew_n = 0
    kurt_n = 0  # excess kurtosis

    print("\nTheoretical moments:")
    print(f"Mean: {mean_n}")
    print(f"Variance: {var_n}")
    print(f"Skewness: {skew_n}")
    print(f"Kurtosis: {kurt_n}")
