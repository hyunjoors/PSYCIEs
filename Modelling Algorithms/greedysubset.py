# Input: number of features F
# matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# vector y of scalar values, with n rows (samples), 1 column
# y[i] is the scalar value of the i-th sample
# Output: vector of selected features S, with F rows, 1 column
# vector thetaS, with F rows, 1 column
# thetaS[0] corresponds to the weight of feature S[0]
# thetaS[1] corresponds to the weight of feature S[1]
# and so on and so forth
import numpy as np
import linreg

def run(F,X,y):
	# Your code goes here
	S = []
	J = [i for i in range(0,len(X[0]))] # no of features
	for f in range(1,F+1):
		# get list of features to iterate on.
		f_iter = []
		for t in J:
			if not (t in S):
				f_iter.append(t)
		best_j = None
		min_error = float("inf")
		for j in f_iter:
			# get X_ = x_{t,sUj}
			S_t = [i for i in S]
			S_t.append(j)
			X_ = X[:,S_t]
			theta_s_u_j = linreg.run(X_,y)

			# use theta_s_u_j to get training error

			error = (0.5) * np.sum((y - np.dot(X_,theta_s_u_j))** 2)
			if error < min_error:
				min_error = error
				best_j = j

		S.append(best_j)

	#thetaS
	X_ = X[:,S]
	thetaS = linreg.run(X_,y)
	return (np.asarray(S).reshape(F,1), thetaS)

# X,y = createlinregdata.run(10,4)
# F = 4
# print run(F,X,y)
# print linreg.run(X,y)
