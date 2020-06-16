from scipy.linalg import svd
import numpy as np
A = np.array([[1,2],
              [1,1],
              [0,0]])
p, s, q = svd(A, full_matrices=False)
print('P=',p)
print('S=',s)
print('Q=',q)
