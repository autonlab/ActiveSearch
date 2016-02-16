from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss

from multiprocessing import Pool

import time
import os, os.path as osp
import csv
import cPickle as pick
import argparse

import activeSearchInterface as ASI
import competitorsInterface as CI
import data_utils as du
import graph_utils as gu
import lapsvmp as LapSVM
import anchorGraph as AG

import IPython

def test_covtype (arg_dict):

	X0 = arg_dict['X0']
	Y0 = arg_dict['Y0']
	Z = arg_dict['Z']
	rL = arg_dict['rL']

	if 'seed' in arg_dict:
		seed = arg_dict['seed']
	else: seed = None
	
	if 'n_init' in arg_dict:
		n_init = arg_dict['n_init']
	else: n_init = 1

	if 'prev' in arg_dict:
		prev = arg_dict['prev']
	else: prev = 0.05

	if 'proj' in arg_dict:
		proj = arg_dict['proj']
	else: proj = False

	if 'save' in arg_dict:
		save = arg_dict['save']
	else: save = False

	verbose=True
	sparse = True
	nr.seed()

	t1 = time.time()
	
	# Changing prevalence of +
	if Y0.sum()/Y0.shape[0] < prev:
		prev = Y0.sum()/Y0.shape[0]
		X,Y = X0,Y0
	else:
		t1 = time.time()
		X,Y,inds = du.change_prev (X0,Y0,prev=prev,return_inds=True)
		Z = Z[inds, :]
		print ('Time taken to change prev: %.2f'%(time.time()-t1))

	strat_frac = 1.0
	if strat_frac < 1.0:
		t1 = time.time()
		X, Y, strat_inds = du.stratified_sample(X, Y, classes=[0,1], strat_frac=strat_frac,return_inds=True)
		Z = Z[strat_inds, :]
		print ('Time taken to stratified sample: %.2f'%(time.time()-t1))
	d,n = X.shape

	# init points
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_init,replace=False)]
	init_labels = {p:1 for p in init_pt}

	t1 = time.time()
	# Kernel AS
	pi = prev
	eta = 0.5
	ASprms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta)
	kAS = ASI.kernelAS (ASprms)
	kAS.initialize(X, init_labels=init_labels)
	print ('KAS initialized.')
	
	# NN AS
	normalize = True
	NNprms = CI.NNParameters(normalize=normalize ,sparse=sparse, verbose=verbose)
	NNAS = CI.averageNNAS (NNprms)
	NNAS.initialize(X, init_labels=init_labels)
	print ('NNAS initialized.')

	# # anchorGraph AS
	gamma = 0.01
	AGprms = CI.anchorGraphParameters(gamma=gamma, sparse=sparse, verbose=verbose)
	AGAS = CI.anchorGraphAS (AGprms)
	AGAS.initialize(Z, rL, init_labels=init_labels)	
	print ('AGAS initialized.')

	hits_K = [n_init]
	hits_NN = [n_init]
	hits_AG = [n_init]

	print ('Time taken to initialize all approaches: %.2f'%(time.time()-t1))
	print ('Beginning experiment.')

	K = 200
	for i in xrange(K):

		print('Iter %i out of %i'%(i+1,K))
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		hits_K.append(hits_K[-1]+Y[idx1])

		idx2 = NNAS.getNextMessage()
		NNAS.setLabelCurrent(Y[idx2])
		hits_NN.append(hits_NN[-1]+Y[idx2])

		idx4 = AGAS.getNextMessage()
		AGAS.setLabelCurrent(Y[idx4])
		hits_AG.append(hits_AG[-1]+Y[idx4])
		print('')
	
	if save:
		if seed is None: 
			seed = -1
		save_results = {'kAS': hits_K,
						'NNAS': hits_NN,
						'AGAS': hits_AG}


		fname = 'expt_seed_%d.cpk'%seed
		if proj:
			dname = osp.join(results_dir, 'main/%.2f/proj/'%(prev*100))
		else:
			dname = osp.join(results_dir, 'main/%.2f/'%(prev*100))
		if not osp.isdir(dname):
			os.makedirs(dname)
		fname = osp.join(dname,fname)
		with open(fname, 'w') as fh: pick.dump(save_results, fh)
	else:
		IPython.embed()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Covtype expts.')
	parser.add_argument('--num_expts', help='number of experiments', default=3, type=int, choices=range(1,11))
	parser.add_argument('--prev', help='prevalence of positive class', default=0.05, type=float)
	parser.add_argument('--proj', help='use projection', action='store_true')
	parser.add_argument('--n_init', help='number of initial positives', default=1, type=int)
	parser.add_argument('--save', help='save results', action='store_true')

	args = parser.parse_args()

	prev = args.prev
	num_expts = args.num_expts
	if prev < 0 or prev > 0.05:
		prev = 0.05
	proj = args.proj
	save = args.save
	n_init = args.n_init
	if n_init < 1: n_init = 1

	data_dir = du.data_dir
	results_dir = osp.join(du.results_dir, 'kdd/covtype/expts')
	## LOAD THE DATA.
	t1 = time.time()
	X0,Y0,classes = du.load_covertype(sparse=True, normalize=False)
	print ('Time taken to load covtype data: %.2f'%(time.time()-t1))

	X0 = du.bias_square_normalize_ft(X0,sparse=True)
	kmeans_fl = osp.join(data_dir, 'covtype_kmeans300_unnormalized.npz')
	Anchors = np.load(kmeans_fl)['arr_0']
	Anchors = du.bias_square_ft(Anchors.T, sparse=False).T
	if proj:
		X0, Y0, L, train_samp = du.project_data3 (X0,Y0,NT=10000)
		Anchors = du.matrix_squeeze(Anchors.dot(L))
		save_proj_file = osp.join(data_dir, 'covtype_proj_mat')
		np.savez(save_proj_file, L=np.array(L), train_samp=train_samp)

		t1 = time.time()
		Z,rL = AG.AnchorGraph(X0, Anchors.T, s=3, flag=1, cn=10, sparse=True, normalized=True)
		print ('Time taken to get AG: %.2f'%(time.time()-t1))

		ag_file = osp.join(data_dir,'/covtype_AG_kmeans300_proj_%.2f'%prev)
		AG.save_AG(ag_file, Z, rL)
	else:
		ag_file = osp.join(data_dir,'covtype_AG_kmeans300.npz')
		Z,rL = AG.load_AG(ag_file)

	# seeds = nr.choice(int(10e6),num_expts,replace=False)
	seeds = range(1, num_expts+1)

	arg_dicts = []
	for s in seeds:
		arg_dicts.append({'X0':X0, 'Y0':Y0, 'Z':Z, 'rL':rL, 'prev':prev, 'proj':proj, 'n_init':n_init, 'seed':s, 'save':save})

	if num_expts == 1:
		print ('Running 1 experiment')
		test_covtype(arg_dicts[0])
	else:
		print ('Running %i experiments'%num_expts)
		pl = Pool(num_expts)
		pl.map(test_covtype, arg_dicts)