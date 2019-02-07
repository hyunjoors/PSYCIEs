# Input: number of bootstraps B
#        numpy matrix X of features, with n rows (samples), d columns (features)
#        numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of B rows, 1 column
import numpy as np
import linreg
def run(B,X,y):
    (n, d) = np.shape(X)
    z = np.zeros((B, 1))
    for i in range(0,B):
        u = [0] * n
        S = set()
        for j in range(0,n):
            k = np.random.randint(0,n)
            u[j] = k
            S.add(k)
        
        T = set(range(0,n)) - S
        thetaHat = linreg.run(X[u], y[u])

        summ = 0
        for t in T:
            summ += (y[t] - np.dot(X[t], thetaHat))**2
        z[i] = (1.0/len(T))*summ
    return z