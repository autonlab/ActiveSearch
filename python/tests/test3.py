from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib.pyplot as plt
import time

import activeSearchInterface as ASI

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def createFakeData (n, r, nt, rcross=10):
	"""
	Builds data set
	"""

	low, high = 0, 50/n
	low_c, high_c = 0, 1/(100*n)

	Xt_t = nr.uniform(low=low, high=high, size=(r,nt))
	Xt_n = nr.uniform(low=low_c, high=high_c, size=(r - rcross, nt))
	Xn_n = nr.uniform(low=low, high=high, size=(r,n-nt))
	Xn_t = nr.uniform(low=low_c, high=high_c, size=(r - rcross, n-nt))

	X = np.c_[np.r_[Xt_t, Xt_n],np.r_[Xn_t, Xn_n]]
	Y = np.array([1]*nt + [0]*(n-nt))

	return X, Y

if __name__ == '__main__':
	pass