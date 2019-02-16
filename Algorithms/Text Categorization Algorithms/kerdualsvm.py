import cvxopt as co
import numpy as np
import K


def run(X, y):
    n = y.shape[0]
    A = -np.identity(n)
    b = np.zeros(n)
    f = -np.ones(n)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = y[i] * y[j] * K.run(X[i, :], X[j, :])
    alpha = np.array(co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'),
                                   co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])
    return alpha
