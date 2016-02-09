from __future__ import division
import time
import os, os.path as osp

import sklearn.cluster as skc
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.sparse as ss, scipy.sparse.linalg as ssl
# import scipy.linalg as slg, scipy.io as sio

import data_utils as du
import anchorGraph as AG
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

def data_kmeans(dataset = 'HIGGS'):

	if dataset == 'HIGGS':
		X,Y,_ = du.load_higgs()
	elif dataset == 'SUSY':
		X,Y,_ = du.load_SUSY()
	elif dataset == 'covtype':
		X,Y,_ = du.load_covertype()

	k = 300
	n_jobs = 10

	t1 = time.time()
	save_file = osp.join(data_dir, '%s_kmeans'%dataset)
	print('Time taken to load covertype data: %.2f\n'%(time.time()-t1))

	Xk = save_kmeans(X.T, save_file, k=k, n_jobs=n_jobs)

def create_AG (dataset = 'covtype', flag=1, s=5, cn=10, normalized=True, k=None):
	if dataset == 'HIGGS':
		X,Y,_ = du.load_higgs()
		Xk = np.load()
	elif dataset == 'SUSY':
		X,Y,_ = du.load_SUSY()
	elif dataset == 'covtype':
		X,Y,_ = du.load_covertype()

	if k is None:
		kmeans_fl = osp.join(data_dir, '%s_kmeans.npz'%dataset)
	else:
		kmeans_fl = osp.join(data_dir, '%s_kmeans_%i.npz'%(dataset,k))
	Anchors = np.load(kmeans_fl)['arr_0']

	Z,rL = AG.AnchorGraph(X.T, Anchors, s=s, flag=flag, cn=cn, sparse=True, normalized=normalized)
	if k is None:
		ag_file = osp.join(data_dir, '%s_AG_kmeans'%dataset)
	else:
		ag_file = osp.join(data_dir, '%s_AG_kmeans%i'%(dataset, k))

	AG.save_AG(ag_file, Z, rL)

if __name__ == '__main__':
	create_AG('covtype')
