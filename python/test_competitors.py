from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
# import matplotlib.pyplot as plt
import time
import os, os.path as osp
import csv
import cPickle as pick
# import sqlparse as sql

import activeSearchInterface as ASI
import similarityLearning as SL

import data_utils as du
import graph_utils as gu

import lapsvmp as LapSVM
import anchorGraph as AG

import IPython

def test_lapsvm ():
	
	n,r = 10000,30
	nt = 50

	# data_dict = sio.loadmat('lapsvm_test.mat')
	# X = data_dict['X']
	# Y = data_dict['Y']
	# Yt = data_dict['Yt']
	X = np.r_[nr.randn(int(n/2),r), (2*nr.randn(n-int(n/2),r)+2)]
	Y = np.array([1]*int(n/2) + [-1]*(n-int(n/2)))

	options = LapSVM.LapSVMOptions()
	options.gamma_I = 1
	options.gamma_A = 1e-5
	options.NN = 6
	options.KernelParam = 0.35
	options.Verbose = True
	options.UseBias = True
	options.UseHinge = True
	options.LaplacianNormalize = False
	options.NewtonLineSearch = False

	Yt = np.zeros(n)
	Yt[:nt] = 1
	Yt[int(n/2):int(n/2)+nt] = -1
	data = LapSVM.LapSVMData(X, Yt, options, sparse=False)

	print('Training LapSVM in the primal with newton steps...\n');
	classifier1 = LapSVM.trainLapSVM(data, options)
	print('It took %f seconds.\n'%classifier1.traintime)

	out1 = np.sign(data.K[:,classifier1.svs].dot(classifier1.alpha)+classifier1.b)
	er1 = 100*(len(data.Y) - np.sum(out1==Y))/len(data.Y)

	# training the classifier
	print('Training LapSVM in the primal with early stopped PCG...\n')
	options.Cg = 1 # PCG
	options.MaxIter = 1000  # upper bound
	options.CgStopType = 1 # 'stability' early stop
	options.CgStopParam = 0.015 # tolerance: 1.5%
	options.CgStopIter = 3 # check stability every 3 iterations
	classifier2 = LapSVM.trainLapSVM(data, options)
	print('It took %f seconds.\n'%classifier2.traintime)

	# computing error rate
	out2 = np.sign(data.K[:,classifier2.svs].dot(classifier2.alpha)+classifier2.b)
	er2 = 100*(len(data.Y) - np.sum(out2==Y))/len(data.Y)

	IPython.embed()


def test_anchorgraph():
	sparse = True

	dat_dir = osp.join(os.getenv('HOME'), 'opt/Anchor_Graph')
	mdat = sio.loadmat(osp.join(dat_dir, 'USPS-MATLAB-train.mat'))
	mdat_labels = sio.loadmat(osp.join(dat_dir, 'usps_label_100.mat'))
	mdat_anchors = sio.loadmat(osp.join(dat_dir, 'usps_anchor_1000.mat'))

	if sparse:
		data = ss.csr_matrix(mdat['samples'])
		anchor = ss.csr_matrix(mdat_anchors['anchor']).T
	else:
		data = mdat['samples']
		anchor = mdat_anchors['anchor'].T

	labels = mdat['labels'].squeeze()
	label_index = mdat_labels['label_index']

	r,n = data.shape
	m = 1000
	s = 3
	cn = 10
	C = labels.max()

	# construct an AnchorGraph(m,s) with kernel weights
	# Z1, rL1 = AnchorGraph(data, anchor, s, 0, cn, sparse)
	# rate0 = np.zeros(20)
	# for i in range(20):
	# 	run_labels = {(li-1):labels[li-1] for li in label_index[i,:]}
	# 	F, A, op = AnchorGraphReg(Z1, rL1, run_labels, C, 0.01, sparse)
	# 	rate0[i] = (op!=labels).sum()/(n-len(run_labels))
	# print('\n The average classification error rate of AGR with kernel weights is %.2f.\n'%(100*np.mean(rate0)))

	# construct an AnchorGraph(m,s) with LAE weights
	Z2, rL2 = AG.AnchorGraph(data, anchor, s, 1, cn, sparse)
	rate = np.zeros(20)
	for i in range(20):
		run_labels = {(li-1):labels[li-1] for li in label_index[i,:]}
		F, A, op = AG.AnchorGraphReg(Z2, rL2, run_labels, C, 0.01, sparse)
		rate[i] = (op!=labels).sum()/(n-len(run_labels))

	print('\n The average classification error rate of AGR with LAE weights is %.2f.\n'%(100*np.mean(rate)))

	IPython.embed()

if __name__ == '__main__':
	# test_lapsvm()
	test_anchorgraph()