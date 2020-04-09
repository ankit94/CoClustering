import numpy as np
from numpy.linalg import inv

#number of data samples
n = 6

#number of runs
m = 3

#number of clusters
c = 3

d = 9

def indicator_matrix(lables, c):
    b = np.arange(1, c, 1)
    return (lables[:, None] == b).astype(int)

def k_means_lables(n):
    k = np.zeros(n)
    return k


#initailize F, G
F = indicator_matrix(k_means_lables(n), c)
G = indicator_matrix(k_means_lables(d), c)

f_term = inv(np.matmul((np.matmul(F.T, np.matmul(F, G.T))), G).T)

R = np.zeros((6,9))
print(R)
s_term = np.matmul(F.T, np.matmul(R, G))
S = np.matmul(f_term, s_term)

W = np.matmul(F, S)

L = np.matmul(S, G.T)

for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        for k in L.shape[0]:
            pass

#access column
# test = np.array([[1, 2], [3, 4], [5, 6]])
# print(test[:,0])
#access row
# print(test[1,:])