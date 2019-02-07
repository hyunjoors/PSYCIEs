import numpy as np

# Input: number of iterations L
# 		number of labels k
# 		matrix X of features, with n rows (samples), d columns (features)
# 			X[i,j] is the j-th feature of the i-th sample
# 		vector y of labels, with n rows (samples), 1 column
# 			y[i] is the label (1 or 2 ... or k) of the i-th sample
# Output: vector theta of d rows, 1 column
#		 vector b of k-1 rows, 1 column
def run(L,k,X,y):
	(n,d)=np.shape(X)
	theta = np.zeros((d, 1))
	b = np.array([ i for i in range(k-1)]).reshape(k-1, 1)
	# b = np.zeros((d,1))
	for iterations in range(0, L):
		for t in range(0, n):
			E = []
			for l in range (k-1):
				s = -1 if (y[t] <= l + 1) else 1

				# print('y: {}, l: {}, s: {}, b: {}'.format(y[t], l + 1, s, b[l]))
				# print((np.dot(X[t],  theta)))
				# print('{}: {}'.format(l, s * ((np.dot(X[t],  theta))[0] - b[l])))

				if ((s * ((np.dot(X[t], theta)) - b[l])) <= 0):
					E.append(l)
			if (len(E) != 0):
				# print(E)
				# perform collective update based on mistakes
				mult = 0
				for index in range(len(E)):
					s = -1 if (y[t] <= E[index] + 1) else 1
					mult = mult + s
				theta = theta + np.array([mult*X[t]]).T

				# update thresholds of each classifier
				for index in range(len(E)):
					s = -1 if (y[t] <= E[index] + 1) else 1
					b[E[index]] = b[E[index]] - s  # change the b[] values for all l in E
					# mult = mult + s  # sum of all stl values of l in E
				# print('SUM OF STUFF:')
				# print(mult)
				# print(X[t])
				# print(np.array([mult*X[t]]).T) 

	return (theta, b)
