import numpy as np
import nocd
import scipy.sparse as sp

loader = np.load("data/facebook_ego/sina.npz", allow_pickle=True)

A, X, Z_gt = loader['A'], loader['X'], loader['Z']

print(type(Z_gt[0][0]))
"""
indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
final = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
print(type(final))
"""

a = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])

final = sp.csr_matrix(a)
print(final)

print(final.nonzero())
