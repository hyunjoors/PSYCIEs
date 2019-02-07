# Input: number of folds k
#        numpy matrix X of features, with n rows (samples), d columns (features)
#        numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
import numpy as np
import linreg

def run(k,X,y):
    (n, d) = np.shape(X)
    z = np.zeros((k, 1))
    for i in range(0,k):
        T = set(range(int(np.floor((n*i)/k)), int(np.floor(((n*(i+1))/k)-1))+1))
        S = set(range(0, n)) - T

        thetaHat = linreg.run(X[list(S)], y[list(S)])
        
        summ = 0
        for t in T:
            summ += (y[t] - np.dot(X[t], thetaHat))**2
        z[i] = (1.0/len(T))*summ
    return z