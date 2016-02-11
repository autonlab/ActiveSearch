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

def data_kmeans(dataset='HIGGS', k=100, n_jobs=10):

	t1 = time.time()
	if dataset == 'HIGGS':
		X,Y,_ = du.load_higgs(normalized=False)
	elif dataset == 'SUSY':
		X,Y,_ = du.load_SUSY(normalized=False)
	elif dataset == 'covtype':
		X,Y,_ = du.load_covertype(normalized=False)
	print('Time taken to load %s data: %.2f\n'%(dataset, time.time()-t1))


	save_file = osp.join(data_dir, '%s_kmeans%i_unnormalized'%(dataset,k))

	Xk = save_kmeans(X.T, save_file, k=k, n_jobs=n_jobs)

def create_AG (dataset = 'covtype', flag=1, s=3, cn=5, normalized=True, k=None):
	t1 = time.time()
	if dataset == 'HIGGS':
		X,Y,_ = du.load_higgs()
	elif dataset == 'SUSY':
		X,Y,_ = du.load_SUSY()
	elif dataset == 'covtype':
		X,Y,_ = du.load_covertype()
	print('Time taken to load %s data: %.2f\n'%(dataset, time.time()-t1))

	if k is None:
		kmeans_fl = osp.join(data_dir, '%s_kmeans.npz'%dataset)
	else:
		kmeans_fl = osp.join(data_dir, '%s_kmeans%i.npz'%(dataset,k))
	Anchors = np.load(kmeans_fl)['arr_0']

	t1 = time.time()
	Z,rL = AG.AnchorGraph(X, Anchors.T, s=s, flag=flag, cn=cn, sparse=True, normalized=normalized)
	print('Time taken to generate AG: %.2f\n'%(time.time()-t1))

	if k is None:
		ag_file = osp.join(data_dir, '%s_AG_kmeans'%dataset)
	else:
		ag_file = osp.join(data_dir, '%s_AG_kmeans%i'%(dataset, k))

	t1 = time.time()
	AG.save_AG(ag_file, Z, rL)
	print('Time taken to save AG: %.2f\n'%(time.time()-t1))

if __name__ == '__main__':
	# create_AG('covtype')
	data_kmeans('covtype')
