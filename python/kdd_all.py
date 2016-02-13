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

def test_dataset_small (arg_dict):
	global X0, Y0, Z0, rL
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

	if 'n' in arg_dict:
		n = arg_dict['n']
	else: n = None

	if 'results_dir' in arg_dict:
		results_dir = arg_dict['results_dir']
	else: results_dir = osp.join(du.results_dir, 'kdd/covtype/expts')

	verbose=True
	sparse = True
	nr.seed()

	t1 = time.time()

	# Changing prevalence of +
	if Y0.sum()/Y0.shape[0] < prev:
		prev = Y0.sum()/Y0.shape[0]
		X,Y,Z = X0,Y0,Z0
	else:
		t1 = time.time()
		X,Y,inds = du.change_prev (X0,Y0,prev=prev,return_inds=True)
		Z = Z0[inds, :]
		print ('Time taken to change prev: %.2f'%(time.time()-t1))
	del X0, Y0, Z0

	if n is None:
		strat_frac = 1.0
	else:
		assert n <= 10000
		strat_frac = n/X.shape[1]
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

	# # lapSVM AS
	relearnT = 1
	LapSVMoptions = LapSVM.LapSVMOptions()
	LapSVMoptions.gamma_I = 1
	LapSVMoptions.gamma_A = 1e-5
	LapSVMoptions.NN = 6
	LapSVMoptions.KernelParam = 0.35
	LapSVMoptions.Verbose = False ## setting this to be false
	LapSVMoptions.UseBias = True
	LapSVMoptions.UseHinge = True
	LapSVMoptions.LaplacianNormalize = False
	LapSVMoptions.NewtonLineSearch = False
	LapSVMoptions.Cg = 1 # PCG
	LapSVMoptions.MaxIter = 1000  # upper bound
	LapSVMoptions.CgStopType = 1 # 'stability' early stop
	LapSVMoptions.CgStopParam = 0.015 # tolerance: 1.5%
	LapSVMoptions.CgStopIter = 3 # check stability every 3 iterations
	LapSVMprms = CI.lapSVMParameters(options=LapSVMoptions, relearnT=relearnT, sparse=False, verbose=verbose)
	LapSVMAS = CI.lapsvmAS (LapSVMprms)
	LapSVMAS.initialize(du.matrix_squeeze(X.todense()), init_labels=init_labels)
	print ('LapSVMAS initialized.')

	# # anchorGraph AS
	gamma = 0.01
	AGprms = CI.anchorGraphParameters(gamma=gamma, sparse=sparse, verbose=verbose)
	AGAS = CI.anchorGraphAS (AGprms)
	AGAS.initialize(Z, rL, init_labels=init_labels)	
	print ('AGAS initialized.')

	hits_K = [n_init]
	hits_NN = [n_init]
	hits_LSVM = [n_init]
	hits_AG = [n_init]

	print ('Time taken to initialize all approaches: %.2f'%(time.time()-t1))
	print ('Beginning experiment.')

	K = 100
	for i in xrange(K):

		print('Iter %i out of %i'%(i+1,K))
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		hits_K.append(hits_K[-1]+Y[idx1])

		idx2 = NNAS.getNextMessage()
		NNAS.setLabelCurrent(Y[idx2])
		hits_NN.append(hits_NN[-1]+Y[idx2])

		idx3 = LapSVMAS.getNextMessage()
		LapSVMAS.setLabelCurrent(Y[idx3])
		hits_LSVM.append(hits_LSVM[-1]+Y[idx3])

		idx4 = AGAS.getNextMessage()
		AGAS.setLabelCurrent(Y[idx4])
		hits_AG.append(hits_AG[-1]+Y[idx4])
		print('')
	
	if save:
		if seed is None: 
			seed = -1
		save_results = {'kAS': hits_K,
						'NNAS': hits_NN,
						'LSVMAS': hits_LSVM,
						'AGAS': hits_AG}

		fname = 'expt_seed_%d.cpk'%seed
		if proj:
			dname = osp.join(results_dir, 'small/%.2f/proj/'%(prev*100))
		else:
			dname = osp.join(results_dir, 'small/%.2f/'%(prev*100))
		if not osp.isdir(dname):
			os.makedirs(dname)
		fname = osp.join(dname,fname)
		with open(fname, 'w') as fh: pick.dump(save_results, fh)
	else:
		IPython.embed()



def test_dataset (arg_dict):
	global X0, Y0, Z0, rL
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

	if 'results_dir' in arg_dict:
		results_dir = arg_dict['results_dir']
	else: results_dir = osp.join(du.results_dir, 'kdd/covtype/expts')

	verbose=True
	sparse = True
	nr.seed()

	t1 = time.time()

	# Changing prevalence of +
	if Y0.sum()/Y0.shape[0] < prev:
		prev = Y0.sum()/Y0.shape[0]
		X,Y,Z = X0,Y0,Z0
	else:
		t1 = time.time()
		X,Y,inds = du.change_prev (X0,Y0,prev=prev,return_inds=True)
		Z = Z0[inds, :]
		print ('Time taken to change prev: %.2f'%(time.time()-t1))
	del X0, Y0, Z0

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

	
	# Kernel AS
	pi = prev
	eta = 0.5
	ASprms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta)
	kAS = ASI.kernelAS (ASprms)
	t1 = time.time()
	kAS.initialize(X, init_labels=init_labels)
	kAS_init =time.time()-t1
	print ('Time taken to initialize KAS approaches: %.2f'%(time.time()-t1))
	
	# NN AS
	
	normalize = True
	NNprms = CI.NNParameters(normalize=normalize ,sparse=sparse, verbose=verbose)
	NNAS = CI.averageNNAS (NNprms)
	t1 = time.time()
	NNAS.initialize(X, init_labels=init_labels)
	NNAS_init =time.time()-t1
	print ('Time taken to initialize NNAS approaches: %.2f'%(time.time()-t1))

	# # anchorGraph AS
	gamma = 0.01
	AGprms = CI.anchorGraphParameters(gamma=gamma, sparse=sparse, verbose=verbose)
	AGAS = CI.anchorGraphAS (AGprms)
	t1 = time.time()
	AGAS.initialize(Z, rL, init_labels=init_labels)	
	AGAS_init =time.time()-t1
	print ('Time taken to initialize AGAS approaches: %.2f'%(time.time()-t1))

	hits_K = [n_init]
	hits_NN = [n_init]
	hits_AG = [n_init]

	# print ('Time taken to initialize all approaches: %.2f'%(time.time()-t1))
	print ('Beginning experiment.')

	elapsed1 = 0
	elapsed2 = 0
	elapsed3 = 0
	K = 5
	for i in xrange(K):

		print('Iter %i out of %i'%(i+1,K))
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		hits_K.append(hits_K[-1]+Y[idx1])
		elapsed1 += kAS.elapsed

		idx2 = NNAS.getNextMessage()
		NNAS.setLabelCurrent(Y[idx2])
		hits_NN.append(hits_NN[-1]+Y[idx2])
		elapsed2 += NNAS.elapsed

		idx4 = AGAS.getNextMessage()
		AGAS.setLabelCurrent(Y[idx4])
		hits_AG.append(hits_AG[-1]+Y[idx4])
		elapsed3 += AGAS.elapsed
		print('')
	
	elapsed1 /= K
	elapsed2 /= K
	elapsed3 /= K

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
	parser = argparse.ArgumentParser(description='KDD expts.')
	parser.add_argument('--dset', help='dataset', default='covtype', type=str, choices=['covtype', 'SUSY', 'HIGGS'])
	parser.add_argument('--etype', help='expt type', default='main', type=str, choices=['main', 'small'])
	parser.add_argument('--num_expts', help='number of experiments', default=3, type=int, choices=range(1,11))
	parser.add_argument('--n', help='number of points for small expt', default=10000, type=int)
	parser.add_argument('--prev', help='prevalence of positive class', default=0.05, type=float)
	parser.add_argument('--proj', help='use projection', action='store_true')
	parser.add_argument('--load_proj', help='load projection', action='store_true')
	parser.add_argument('--n_init', help='number of initial positives', default=1, type=int)
	parser.add_argument('--save', help='save results', action='store_true')

	args = parser.parse_args()

	dset = args.dset
	etype = args.etype
	prev = args.prev
	num_expts = args.num_expts

	n = args.n
	if n < 0 or n > 10000:
		n = 10000
	if prev < 0 or prev > 0.05:
		prev = 0.05

	proj = args.proj
	load_proj = args.load_proj
	save = args.save
	n_init = args.n_init
	if n_init < 1: n_init = 1

	data_dir = du.data_dir
	results_dir = osp.join(du.results_dir, 'kdd/%s/expts'%dset)
	## LOAD THE DATA.
	t1 = time.time()
	if dset == 'covtype':
		X0,Y0,classes = du.load_covertype(sparse=True, normalize=False)
	elif dset == 'SUSY':
		X0,Y0,classes = du.load_SUSY(sparse=True, normalize=False)
	else:
		X0,Y0,classes = du.load_higgs(sparse=True, normalize=False)
	print ('Time taken to load %s data: %.2f'%(dset, time.time()-t1))

	if dset == 'covtype':
		X0 = du.bias_square_normalize_ft(X0,sparse=True)
		kmeans_fl = osp.join(data_dir, 'covtype_kmeans300_unnormalized.npz')
		Anchors = np.load(kmeans_fl)['arr_0']
		Anchors = du.bias_square_normalize_ft(Anchors.T, sparse=False).T
	else:
		X0 = du.bias_normalize_ft(X0,sparse=True)
		kmeans_fl = osp.join(data_dir, '%s_kmeans100_unnormalized.npz'%dset)
		Anchors = np.load(kmeans_fl)['arr_0']
		Anchors = du.bias_normalize_ft(Anchors.T, sparse=False).T
		

	if proj:
		if load_proj:
			proj_file = osp.join(data_dir, '%s_proj_mat_%.4f.npz'%(dset, prev))
			
			if dset == 'covtype':			
				ag_file = osp.join(data_dir,'%s_AG_kmeans300_proj_%.4f.npz'%(dset, prev))
			else:
				ag_file = osp.join(data_dir,'%s_AG_kmeans100_proj_%.4f.npz'%(dset, prev))

			projdat = np.load(proj_file)
			L = projdat['L']
			train_samp = projdat['train_samp']
			X0, Y0 = du.apply_proj(X0,Y0,L,train_samp)
			# Anchors = du.matrix_squeeze(Anchors.dot(L))

			Z0,rL = AG.load_AG(ag_file)
		else:
			X0, Y0, L, train_samp = du.project_data3 (X0,Y0,NT=10000)
			Anchors = du.matrix_squeeze(Anchors.dot(L))

			save_proj_file = osp.join(data_dir, '%s_proj_mat_%.4f'%(dset, prev))
			np.savez(save_proj_file, L=np.array(L), train_samp=train_samp)

			if dset == 'covtype':
				ag_file = osp.join(data_dir,'%s_AG_kmeans300_proj_%.4f'%(dset, prev))
				Z0,rL = AG.AnchorGraph(X0, Anchors.T, s=3, flag=1, cn=10, sparse=True, normalized=True)
			else:
				ag_file = osp.join(data_dir,'%s_AG_kmeans100_proj_%.4f'%(dset, prev))
				Z0,rL = AG.AnchorGraph(X0, Anchors.T, s=2, flag=1, cn=5, sparse=True, normalized=True)
			print ('Time taken to get AG: %.2f'%(time.time()-t1))

			AG.save_AG(ag_file, Z0, rL)
	else:
		if dset == 'covtype':
			ag_file = osp.join(data_dir,'covtype_AG_kmeans300.npz')
		else:
			ag_file = osp.join(data_dir,'%s_AG_kmeans100.npz'%dset)
		Z0,rL = AG.load_AG(ag_file)

	# seeds = nr.choice(int(10e6),num_expts,replace=False)
	seeds = range(1, num_expts+1)

	arg_dicts = []
	for s in seeds:
		# arg_dicts.append({'X0':X0, 'Y0':Y0, 'Z0':Z0, 'rL':rL, 'prev':prev, 'proj':proj, 'n_init':n_init, 'seed':s, 'save':save, 'results_dir':results_dir})
		arg_dicts.append({'prev':prev, 'proj':proj, 'n_init':n_init, 'n':n, 'seed':s, 'save':save, 'results_dir':results_dir})

	test_func = {'main':test_dataset, 'small':test_dataset_small}[etype]

	if num_expts == 1:
		print ('Running 1 experiment')
		test_func(arg_dicts[0])
	else:
		print ('Running %i experiments'%num_expts)
		pl = Pool(num_expts)
		pl.map(test_func, arg_dicts)