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

data_dir = os.getenv('AS_DATA_DIR')

def save_kmeans (X, save_file=None, k=300, max_iter=300, n_jobs=4, verbose=100, precompute_distances=True):
	# perform kmeans clustering on X and save it

	km = skc.KMeans(n_clusters=k, max_iter=max_iter, n_jobs=n_jobs, verbose=verbose, precompute_distances=precompute_distances)
	t1 = time.time()
	km.fit(X)
	print('Time taken to cluster: %.2fs'%(time.time() - t1))

	Xk = km.cluster_centers_
	if save_file is not None:
		np.savez(save_file, Xk)
	else:
		return Xk

def covtype_kmeans():

	k = 300
	n_jobs = 4
	save_file = osp.join(data_dir, 'covtype_kmeans')

	t1 = time.time()
	X,Y,_ = du.load_covertype()
	print('Time taken to load covertype data: %.2f\n'%(time.time()-t1))

	Xk = save_kmeans(X.T, save_file, k=k, n_jobs=n_jobs)

def HIGGS_kmeans():

	k = 300
	n_jobs = 
	save_file = osp.join(data_dir, 'HIGGS_kmeans')

	t1 = time.time()
	X,Y,_ = du.load_higgs()
	print('Time taken to load HIGGS data: %.2f\n'%(time.time()-t1))

	Xk = save_kmeans(X.T, save_file, k=k, n_jobs=n_jobs)

if __name__ == '__main__':
	covtype_kmeans()
