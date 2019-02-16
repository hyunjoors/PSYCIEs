# Input: number of iterations L
#        numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#            y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of weights, with L rows, 1 column
#         numpy vector theta of feature indices, with L rows, 1 column
import numpy as np

def sgn(z):
    return 1 if z>0 else -1

def run(L,X,y):
    (n,d) = np.shape(X)
    W = []
    theta = [0]*L
    alpha = [0]*L
    for t in range(0,n):
        W.append(1.0/n)
    for r in range(0,L):
        epsilon = np.Inf
        for j in range(0,d):
            summ = 0
            for t in range(0,n):
                summ += W[t]*y[t]*sgn(X[t][j])
            summ = -1*summ
            if(summ < epsilon):
                epsilon = summ
                theta[r] = j
        epsilon = np.min([0.99, np.max([-0.99, epsilon])])
        alpha[r] = 0.5*np.log((1.0-epsilon)/(1.0+epsilon))
        for t in range(0,n):
            j = theta[r]
            W[t] = W[t]*np.exp(-1*alpha[r]*y[t]*sgn(X[t][theta[r]]))
        Z = 0
        for t in range(0,n):
            Z += W[t]
        for t in range(0,n):
            W[t] = W[t]/Z

    alpha = np.reshape(alpha, (L, 1))
    theta = np.reshape(theta, (L, 1))
    return (alpha, theta)