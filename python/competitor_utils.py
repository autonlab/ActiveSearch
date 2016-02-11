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
	save_file = osp.join(data_dir, '%s_kmeans%i_unnormalized'%(dataset,k))
	if dataset == 'HIGGS':
		X,Y,_ = du.load_higgs(normalize=False)
	elif dataset == 'SUSY':
		X,Y,_ = du.load_SUSY(normalize=False)
	elif dataset == 'covtype':
		X,Y,_ = du.load_covertype(normalize=False)
	print('Time taken to load %s data: %.2f\n'%(dataset, time.time()-t1))


	Xk = save_kmeans(X.T, save_file, k=k, n_jobs=n_jobs)

def create_AG (dataset = 'covtype', flag=1, s=3, cn=5, normalized=True, k=100, ft=None, proj=False):
	t1 = time.time()
	if dataset == 'HIGGS':
		X,Y,_ = du.load_higgs(normalized=False)
	elif dataset == 'SUSY':
		X,Y,_ = du.load_SUSY(normalized=False)
	elif dataset == 'covtype':
		X,Y,_ = du.load_covertype(normalized=False)
	print('Time taken to load %s data: %.2f\n'%(dataset, time.time()-t1))

	if k is None:
		kmeans_fl = osp.join(data_dir, '%s_kmeans_unnormalized.npz'%dataset)
	else:
		kmeans_fl = osp.join(data_dir, '%s_kmeans%i_unnormalized.npz'%(dataset,k))
	Anchors = np.load(kmeans_fl)['arr_0']
	
	if ft is not None:
		X = ft(X, sparse=True)
		Anchors = ft(Anchors.T, sparse=False).T

	if proj:
		N = 10000
		random_coeff = 0.01 if dataset is 'covtype' else 0
		L, train_samp = du.project_data(X,Y,NT=N,random_coeff=random_coeff, sparse=True)
		rem_inds = np.ones(X.shape[1]).astype(bool)
		rem_inds[train_samp] = False

		X = ss.csc_matrix(L.T.dot(X[:,rem_inds]))
		Y = Y[rem_inds]
		Anchors = Anchors.dot(L)

	t1 = time.time()
	# Measuring "distance" only based on dot-product
	Z,rL = AG.AnchorGraph(X, Anchors.T, s=s, flag=flag, cn=cn, sparse=True, normalized=True)
	print('Time taken to generate AG: %.2f\n'%(time.time()-t1))

	if proj:
		ag_file = osp.join(data_dir, '%s_AG_kmeans%i_proj'%(dataset, k))
	else:
		ag_file = osp.join(data_dir, '%s_AG_kmeans%i'%(dataset, k))

	t1 = time.time()
	AG.save_AG(ag_file, Z, rL)
	print('Time taken to save AG: %.2f\n'%(time.time()-t1))

if __name__ == '__main__':
	create_AG('covtype', s=3, cn=10, normalized=True, k=300, ft=du.bias_square_ft, proj=False)
	# create_AG('covtype', s=3, cn=10, normalized=True, k=300, ft=du.bias_square_ft, proj=True)
	# data_kmeans('covtype', k=300)

	# create_AG('SUSY', s=3, cn=5, normalized=True, k=100, ft=du.bias_normalize_ft, proj=False)
	# create_AG('SUSY', s=3, cn=5, normalized=True, k=100, ft=du.bias_normalize_ft, proj=True)
	# data_kmeans('SUSY', k=100)

	# create_AG('HIGGS', s=3, cn=5, normalized=True, k=100, ft=du.bias_square_ft, proj=False)
	# create_AG('HIGGS', s=3, cn=10, normalized=True, k=100, ft=du.bias_square_ft, proj=True)
	# data_kmeans('HIGGS', k=100)