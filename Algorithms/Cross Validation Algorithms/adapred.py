# Input: numpy vector alpha of weights, with L rows, 1 column
#        numpy vector theta of feature indices, with L rows, 1 column
#        numpy vector x of d rows, 1 column
# Output: label (+1 or -1)
import numpy as np
def sgn(z):
    return 1 if z>0 else -1

def run(alpha,theta,x):
    L, d = np.shape(alpha)
    summ = 0.0
    for r in range(0,L):
        summ += alpha[r]*sgn(x[int(theta[r])])
    return sgn(summ)