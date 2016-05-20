#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import os, os.path as osp
import time
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.linalg as slg, scipy.spatial.distance as ssd, scipy.sparse as ss
import scipy.io as sio
#import matplotlib.pyplot as plt, matplotlib.cm as cm

import IPython # debugging

import activeSearchInterface as ASI
import adaptiveActiveSearch as AAS
import similarityLearning as SL
import dataUtils as du

np.set_printoptions(suppress=True, precision=5, linewidth=100)
#results_dir = osp.join(os.getenv('HOME'), 'Research/ActiveSearch/Results')
results_dir = os.getenv('AS_NIPS_RDIR')

def polarToCartesian (r, theta):
	return r*np.array([np.cos(theta), np.sin(theta)])

def cartesianToPolas (x, y):
	return np.array([nlg.norm([x,y]), np.arctan(y,x)])

def createSwissRolls (npts = 500, prev = 0.5, c = 1.0, nloops = 1.5, var1 = 0.05, var2 = 0.05, shuffle=False):
	# npts 		-- number of points overall
	# prev 		-- prevalence of positive class
	# c 		-- r = c*theta
	# nloops	-- number of loops of swiss roll
	# var 		-- variance of 0-mean gaussian noise along the datapoints
	# shuffle	-- shuffle points or keep them grouped as 1/0

	std1 = np.sqrt(var1)
	std2 = np.sqrt(var2)
	n1 = int(prev*npts);
	n2 = npts-n1

	angle_range1 = np.linspace(np.pi/2, 2*nloops*np.pi, n1)
	angle_range2 = np.linspace(np.pi/2, 2*nloops*np.pi, n2)

	X = np.empty([npts,2])
	Y = np.array(n1*[1] + n2*[0])

	for i in xrange(n1):
		a = angle_range1[i]
		X[i,:] = polarToCartesian(a*c, a) + nr.randn(1,2)*std1
	for i in xrange(n2):
		a = angle_range2[i]
		X[i+n1,:] = polarToCartesian(a*c, a+np.pi) + nr.randn(1,2)*std2

	if shuffle:
		shuffle_inds = nr.permutation(npts)
		X = X[shuffle_inds,:]
		Y = Y[shuffle_inds]

	return X,Y

def plotData(X, Y, f=None, labels=None, thresh=None, block=False, fid=None):

	if fid is not None:
		fig = plt.figure(fid)
	plt.clf()

	if f is None:
		pos_inds = (Y==1).nonzero()[0]
		neg_inds = (Y==0).nonzero()[0]

		plt.scatter(X[pos_inds,0], X[pos_inds,1], color='b', label='positive')
		plt.scatter(X[neg_inds,0], X[neg_inds,1], color='r', label='negative')
		
	else:
		# assert thresh is not None
		assert labels is not None

		pos_inds = (labels==1).nonzero()[0]
		plt.scatter(X[pos_inds,0], X[pos_inds,1], color='b', label='positive', marker='x', linewidth=2)
		neg_inds = (labels==0).nonzero()[0]	
		plt.scatter(X[neg_inds,0], X[neg_inds,1], color='r', label='negative', marker='x', linewidth=2)

		plt.set_cmap('RdBu')
		rest_inds = (labels==-1).nonzero()[0]
		colors = cm.RdBu(f)
		plt.scatter(X[rest_inds,0], X[rest_inds,1], color=colors, label='unlabeled', linewidth=1)

	if fid is not None:
		plt.title('ID: %i'%fid)
	# plt.legend()
	plt.show(block=block)
	plt.pause(0.001)
	# time.sleep(0.5)

def createLinearGraph (X):
		return np.dot(X,X.T)

def createPolyGraph (X,d,c):
		return np.power(np.dot(X,X.T)+c,d)

def createEpsilonGraph (X, eps=1, kernel='rbf', gamma=1):
	## Creates an epsilon graph as follows:
	## create with edges between points with distance < eps
	## edge weights are given by kernel
	
	if kernel not in ['rbf']:
		raise NotImplementedError('This function does not support %s kernel.'%kernel)

	dists = ssd.cdist(X,X)
	eps_neighbors = dists < eps

	if kernel == 'rbf':
		A = eps_neighbors*np.exp(-gamma*dists)

	return A

def testSwissRolls ():
	npts = 500
	prev = 0.5
	c = 1.0
	nloops = 1.5
	var = 0.05
	shuffle = False
	
	X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var, shuffle=shuffle)
	plotData(X,Y)


def testAS_run (A1,A2,X,Y,prev,verbose,initp_pt,initn_pt,K):

	## Initialize naiveAS
	pi = prev
	sparse = False

	prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi)

	np_init = len(initp_pt)
	nn_init = len(initn_pt)
	n_init = np_init + nn_init
	init_labels = {p:1 for p in initp_pt}
	for p in initn_pt: init_labels[p] = 0

	lAS1 = ASI.naiveAS (prms)
	lAS1.initialize(A1, init_labels)
	lAS2 = ASI.naiveAS (prms)
	lAS2.initialize(A2, init_labels)

	#plotData(X, None, lAS1.f, lAS1.labels, fid=0)
	#plotData(X, None, lAS2.f, lAS2.labels, fid=1)

	hits1 = [n_init]
	hits2 = [n_init]
	#K = 200
	for i in xrange(K):

		if verbose: print('Iter %i out of %i'%(i+1,K))
		idx1 = lAS1.getNextMessage()
		idx2 = lAS2.getNextMessage()
		lAS1.setLabelCurrent(Y[idx1])
		lAS2.setLabelCurrent(Y[idx2])
		hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])

		#plotData(X, None, lAS1.f, lAS1.labels, fid=0)
		#plotData(X, None, lAS2.f, lAS2.labels, fid=1)

		if verbose: print('')
	#IPython.embed()
	return (hits1,hits2)


def KTA(y,A):
	n = A.shape[0]
	return np.dot(2*y-1,np.dot(A,2*y-1))/(n*nlg.norm(A,'fro'))

def KTA2(y,A):
	n = A.shape[0]
	D = np.squeeze(A.sum(1))
	M = np.dot(np.diag(1./D),A)
	return np.dot(y,np.dot(M,y))/n

def KTA_AS(y,A):
	n = A.shape[0]
	w0 = 1/n
	B = (1+w0)
	D = np.squeeze(A.sum(1))
	M = np.diag(np.squeeze(B*D))-A
	Mhat = np.dot(nlg.inv(M),np.diag(w0*D))
	return KTA(y,Mhat)

def LinInterpKernel(x,Y,A,B):
	Anorm, Bnorm = nlg.norm(A,'fro'), nlg.norm(B,'fro')
	a,b,c = KTA(Y,A), KTA(Y,B), np.trace(np.dot(A.T,B))/Anorm/Bnorm
	A_prime = A if a>b else B
	theta1 = -(a*b-c*x*x)/(a*a-x*x)+np.sqrt((a*b-c*x*x)**2-(a*a-x*x)*(b*b-x*x))/(a*a-x*x)
	theta2 = -(a*b-c*x*x)/(a*a-x*x)-np.sqrt((a*b-c*x*x)**2-(a*a-x*x)*(b*b-x*x))/(a*a-x*x)
	if theta1>=0:
	  theta=theta1
	elif theta2>=0:
	  theta=theta2
	else:
	  return A_prime
	alpha = 1/(Anorm/theta/Bnorm+1)
	beta=1-alpha
	A_prime = alpha*A+beta*B
	return A_prime

def OptLinInterpKernel(Y,A,B):
	Anorm, Bnorm = nlg.norm(A,'fro'), nlg.norm(B,'fro')
	a,b,c = KTA(Y,A), KTA(Y,B), np.trace(np.dot(A.T,B))/Anorm/Bnorm
	theta = (b*c-a)/(a*c-b)
	if theta>=0:
	  alpha = 1/(Anorm/theta/Bnorm+1)
	else:
	  alpha = 1 if a>b else 0
	beta=1-alpha
	A_prime = alpha*A+beta*B
	return A_prime

def runTests (prev,K,verbose=False):
	## Create swiss roll data
	npts = 1000
	#prev = 0.05
	c = 1
	nloops = 1.5
	var1 = 0.1
	var2 = 2.0
	shuffle = False
	eps = np.inf
	gamma = 10
	
	X,Y = createSwissRolls(npts=npts, prev=prev, c=c, nloops=nloops, var=var1,var2=var2, shuffle=shuffle)

	k = 10
	sigma = 'local-scaling'
	A_nn,sigma1 = du.generate_nngraph (X, k=k, sigma=sigma)
	A_lin = createLinearGraph(X)
	A_poly = createPolyGraph(X,4,1)
	A_eps = createEpsilonGraph (X, eps=eps, gamma=gamma)
	# Perform AEW 
	max_iter = 100
	param = SL.MSALPParameters(k=k, sigma=sigma, max_iter=max_iter)
	A_aew,A_nn0 = SL.AEW(X,param)#,verbose)
	A_ideal = 0.5*(np.outer(2*Y-1,2*Y-1)+1)
	A_rand = np.random.random((npts,npts))

	#plotData(X, Y)

	KTA_list = {}
	KTA_list['ideal']=[KTA(Y,A_ideal), KTA2(Y,A_ideal), KTA_AS(Y,A_ideal)]
	KTA_list['rand']=[KTA(Y,A_rand), KTA2(Y,A_rand), KTA_AS(Y,A_rand)]
	KTA_list['lin']=[KTA(Y,A_lin), KTA2(Y,A_lin), KTA_AS(Y,A_lin)]
	KTA_list['poly']=[KTA(Y,A_poly), KTA2(Y,A_poly), KTA_AS(Y,A_poly)]
	KTA_list['nn']=[KTA(Y,A_nn), KTA2(Y,A_nn), KTA_AS(Y,A_nn)]
	KTA_list['eps']=[KTA(Y,A_eps), KTA2(Y,A_eps), KTA_AS(Y,A_eps)]
	KTA_list['aew']=[KTA(Y,A_aew), KTA2(Y,A_aew), KTA_AS(Y,A_aew)]
	A_prime = OptLinInterpKernel(Y,A_nn,A_aew) #LinInterpKernel(KTA(Y,A_eps)*1.05,Y,A_ideal,A_eps)
	KTA_list['convex']=[KTA(Y,A_prime), KTA2(Y,A_prime), KTA_AS(Y,A_prime)]

	np_init, nn_init = 1,1
	initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
	initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]

	h_rand,h_ideal = testAS_run (A_rand,A_ideal,X,Y,prev,verbose,initp_pt,initn_pt,K)
	h_lin,h_poly = testAS_run (A_lin,A_poly,X,Y,prev,verbose,initp_pt,initn_pt,K)
	h_nn,h_eps = testAS_run (A_nn,A_eps,X,Y,prev,verbose,initp_pt,initn_pt,K)
	h_prime,h_aew = testAS_run (A_prime,A_aew,X,Y,prev,verbose,initp_pt,initn_pt,K)        
	h_bandit, h_stochweight = testMultipleKernelAS_run([A_rand,A_lin,A_poly,A_nn,A_eps,A_prime],X,Y,prev,verbose,initp_pt,initn_pt,K)
	hits = {}
	hits['ideal']=h_ideal
	hits['rand']=h_rand
	hits['lin']=h_lin
	hits['poly']=h_poly
	hits['nn']=h_nn
	hits['eps']=h_eps
	hits['aew']=h_aew
	hits['convex']=h_prime
	hits['bandit']=h_bandit
	hits['stochweight']=h_stochweight
	return {'hits':hits, 'KTA':KTA_list}

def testMultipleKernelAS_run (As,X,Y,prev,verbose,initp_pt,initn_pt,K):

	## Initialize naiveAS
	pi = prev
	sparse = False

	prms = ASI.Parameters(sparse=sparse, verbose=verbose, pi=pi)

	np_init = len(initp_pt)
	nn_init = len(initn_pt)
	n_init = np_init + nn_init
	initp_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]), np_init, replace=False)]
	initn_pt = (Y==0).nonzero()[0][nr.choice(len(Y.nonzero()[0]), nn_init, replace=False)]
	init_labels = {p:1 for p in initp_pt}
	for p in initn_pt: init_labels[p] = 0

	#K = 200
	nB = len(As) #2 # number of bandits/experts
	gamma = 0.1
	beta = 0.0 # leave this as 0
	exp3params = AAS.EXP3Parameters (gamma = gamma, T=K, nB=nB, beta=beta)
	rwmparams = AAS.RWMParameters (gamma = gamma, T=K, nB=nB)

	mAS1 = AAS.EXP3NaiveAS (prms, exp3params)
	mAS1.initialize(As, init_labels)
	mAS2 = AAS.RWMNaiveAS (prms, rwmparams)
	mAS2.initialize(As, init_labels)

	#plotData(X, None, mAS1.f, mAS1.labels, fid=0)
	#plotData(X, None, mAS2.f, mAS2.ASexperts[0].labels, fid=1)

	hits1 = [n_init]
	hits2 = [n_init]
	for i in xrange(K):

		if verbose: print('Iter %i out of %i'%(i+1,K))
		idx1 = mAS1.getNextMessage()
		idx2 = mAS2.getNextMessage()
		#print idx1, idx2
		mAS1.setLabelCurrent(Y[idx1])
		mAS2.setLabelCurrent(Y[idx2])
		hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])

		# plotData(X, None, mAS1.f, mAS1.labels, fid=0)
		#plotData(X, None, mAS2.f, mAS2.ASexperts[0].labels, fid=1)

		if verbose: print('')

	#IPython.embed()
	return (hits1,hits2)

def RunAllTests():
	N = 2
	sums = {'hits':{}, 'KTA':{}}
	sumsqr = {'hits':{}, 'KTA':{}}
	K = 10
	alpha = 0.05
	for expt in xrange(N): 
	  print('Running experiment %i'%(expt+1))
	  out = runTests(alpha,K)
	  for h in out['hits'].keys():
		if h not in sums['hits']: sums['hits'][h] = [0]*K
		if h not in sumsqr['hits']: sumsqr['hits'][h] = [0]*K
		for i in xrange(K):
		  sums['hits'][h][i] += out['hits'][h][i]
		  sumsqr['hits'][h][i] += out['hits'][h][i]**2
	  for h in out['KTA'].keys():
		if h not in sums['KTA']: sums['KTA'][h] = [0]*3
		if h not in sumsqr['KTA']: sumsqr['KTA'][h] = [0]*3
		for i in xrange(len(out['KTA'][h])):
		  sums['KTA'][h][i] += out['KTA'][h][i]
		  sumsqr['KTA'][h][i] += out['KTA'][h][i]**2
	f = open(osp.join(results_dir,'results_hits%.2f.csv'%alpha),'w')
	f.write('iteration')
	for h in sums['hits'].keys():
	  f.write(','+h+'_mn,'+h+'_sd')
	f.write('\n')
	for i in xrange(K):
	  f.write(str(i+1))
	  for h in sums['hits'].keys():
		mn = float(sums['hits'][h][i])/float(N)
		sd = np.sqrt((float(sumsqr['hits'][h][i])-2*float(sums['hits'][h][i])*mn+float(N)*mn**2)/float(N-1))
		f.write(','+str(mn)+','+str(sd))
	  f.write('\n')
	f.close()
	f = open(osp.join(results_dir,'results_ktas%.2f.csv'%alpha),'w')
	f.write('iteration')
	for h in sums['KTA'].keys():
	  f.write(','+h+'_mn,'+h+'_sd')
	f.write('\n')
	for i in xrange(3):
	  f.write(str(i+1))
	  for h in sums['KTA'].keys():
		mn = float(sums['KTA'][h][i])/float(N)
		sd = np.sqrt((float(sumsqr['KTA'][h][i])-2*float(sums['KTA'][h][i])*mn+float(N)*mn**2)/float(N-1))
		f.write(','+str(mn)+','+str(sd))
	  f.write('\n')
	f.close()

if __name__ == '__main__':
	RunAllTests()
