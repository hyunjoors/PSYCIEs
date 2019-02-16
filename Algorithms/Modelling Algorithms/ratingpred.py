import numpy as np

# Input: number of labels k
# 		vector theta of d rows, 1 column
# 		vector b of k-1 rows, 1 column
# 		vector x of d rows, 1 column
# Output: label (1 or 2 ... or k)
def run(k,theta,b,x):
	compare = (np.dot(x, theta))
#	print(compare)

	for index in range(len(b)):
		# print(b[index])
		if (compare <= b[index]):
			return index + 1
	return k