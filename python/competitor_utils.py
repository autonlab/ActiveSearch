from __future__ import division
import time
import os, os.path as osp

import sklearn.cluster as skc
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.sparse as ss, scipy.sparse.linalg as ssl
# import scipy.linalg as slg, scipy.io as sio

import data_utils as du
# import graph_utils as gu

import IPython

def save_kmeans (X, save_file=None, k=300, max_iter=300, n_jobs=4, verbose=100):
	# perform kmeans clustering on X and save it

	km = skc.KMeans(n_clusters=k, max_iter=max_iter, n_jobs=n_jobs, verbose=verbose)
	t1 = time.time()
	km.fit(X)
	print('Time taken to cluster: %.2fs'%(time.time() - t1))

	Xk = km.cluster_centers_
	if save_file is not None:
		np.savez(save_file, Xk)
	else:
		return Xk


