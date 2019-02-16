import numpy as np
import cvxopt as co

# Input: number of labels k
# 		matrix X of features, with n rows (samples), d columns (features)
# 			X[i,j] is the j-th feature of the i-th sample
# 		vector y of labels, with n rows (samples), 1 column
# 			y[i] is the label (1 or 2 ... or k) of the i-th sample
# Output: vector theta of d rows, 1 column
# 		vector b of k-1 rows, 1 column
def run(k,X,y):
	(n,d)=np.shape(X)

	H = np.zeros((d+k-1, d+k-1))
	for i in range(d):
		for j in range(d):
			if (i == j):
				H[i][j] = 1

	f = np.zeros((d+k-1, 1))

	A1 = np.zeros((n*(k-1), d))
	A2 = np.zeros((n*(k-1), k-1))
	for multiple in range (n):
		for row in range (k-1):
			for column in range (d):
				t = multiple
				l = row
				s = -1 if (y[t] <= l + 1) else 1
				value = -s * X[multiple][column]
				A1[row + ((k-1)*multiple)][column] = value
				# if (row == 1): 
				# 	print("how we got s: is {} <= {}? if so, -1. s: {}".format(y[t],l,s))
				# 	print("how we got value: {} * -({}) = {}".format(X[multiple][column], s, value))
	for multiple in range (n):
		for row in range (k-1):
			for column in range (k-1):
				if (row == column):
					t = multiple
					l = row
					s = -1 if (y[t] <= l + 1) else 1
					A2[row + ((k-1)*multiple)][column] = s

	A3 = np.zeros(((k-2), d))
	A4 = np.zeros(((k-2), k-1))
	for row in range(k-2):
		for column in range(k-1):
			if (row == column):
				A4[row][column] = 1
				A4[row][column + 1] = -1

	A = np.concatenate((np.concatenate((A1,A2), axis=1), np.concatenate((A3,A4), axis=1)), axis= 0)

	c = np.negative(np.ones((n * (k-1), 1)))
	c = np.concatenate((c, np.zeros((k-2, 1))), axis=0)

#	robust = False
#	if (robust):
#		print(H)
#		print(f)
#		print(A)
#		print(c)
#	else:
#		co.solvers.options['show_progress'] = False

	z = np.array(co.solvers.qp(co.matrix(H,tc='d'),co.matrix(f,tc='d'),
		co.matrix(A,tc='d'),co.matrix(c,tc='d'))['x'])

	theta = z[:d]
	b = z[d:]
	return (theta, b)