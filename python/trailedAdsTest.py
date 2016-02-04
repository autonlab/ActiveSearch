from __future__ import division
import sklearn.feature_extraction.text as sft
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib, matplotlib.pyplot as plt

import time
import csv
import os, os.path as osp
import IPython

import data_utils as du
import activeSearchInterface as ASI

data_dir = '/usr0/home/sibiv/Research/Data/ActiveSearch/Jarod/trailed ads classification Jan 2016'

def load_training_ad_data (sparse=True):
	Xfh = open(osp.join(data_dir, 'observations.csv'))
	Yfh = open(osp.join(data_dir, 'labels.csv'))

	Xcsv = csv.reader(Xfh)
	Ycsv = csv.reader(Yfh)

	words = Xcsv.next()
	r = len(words)

	classes = []
	if sparse:
		Y = []
		rows = []
		cols = []
		sdat = []

		c = 0
		for xline, yline in zip(Xcsv, Ycsv):
			y = int(yline[0])
			Y.append(y)
			if y not in classes: classes.append(y)

			xvec = np.array(xline).astype(float)
			xcol = xvec.nonzero()[0].tolist()

			rows.extend(xcol)
			cols.extend([c]*len(xcol))
			sdat.extend(xvec[xcol].tolist())

			c += 1

		X = ss.csr_matrix((sdat, (rows, cols)), shape=(r, c))

	else:

		X = []
		Y = []
		for xline, yline in zip(Xcsv, Ycsv):
			y = int(yline[0])
			Y.append(y)
			if y not in classes: classes.append(y)

			X.append(np.asarray(xline).astype(float))

		X = np.asarray(X).T

	Xfh.close()
	Yfh.close()

	Y = np.asarray(Y)

	return X, Y, classes

def load_test_ad_data (sparse=True):
	Xfh = open(osp.join(data_dir, 'unlabeled.csv'))
	Xcsv = csv.reader(Xfh)

	words = Xcsv.next()
	r = len(words)

	if sparse:
		rows = []
		cols = []
		sdat = []

		c = 0
		for xline in Xcsv:

			xvec = np.array(xline).astype(float)
			xcol = xvec.nonzero()[0].tolist()

			rows.extend(xcol)
			cols.extend([c]*len(xcol))
			sdat.extend(xvec[xcol].tolist())

			c += 1

		X = ss.csr_matrix((sdat, (rows, cols)), shape=(r, c))

	else:

		X = []
		for xline in Xcsv:
			X.append(np.asarray(xline).astype(float))

		X = np.asarray(X).T

	fn.close()

	return X

def tfidf_from_word_count(X_counts):
	## X_counts -- r x n sparse matrix

	tfidf = sft.TfidfTransformer()
	return tfidf.fit_transform(X_counts.T).T

def plot_expts (hits, prev=0, max_possible=None, ind_expts=True, title='', save=False):

	num_exp, max_iter = hits.shape

	itr = range(max_iter)
	chance = (np.array(itr)*prev).tolist()
	if max_possible is None:
		ideal = (np.array(itr)+1).tolist()
	else:
		ideal = range(1, max_possible+1) + [max_possible+1]*(max_iter-max_possible)

	mean_hits = hits.mean(axis=0).squeeze()
	std_hits = hits.std(axis=0).squeeze()
	max_hits = hits.max(axis=1).squeeze()
	# mean2_hits = hits.mean(axis=1).squeeze()
	ax = plt.subplot()

	color = 'b'
	if ind_expts:
		for run in range(num_exp):
			ax.plot(itr, hits[run, :], color=color, alpha=0.2, linewidth=2)
	else:
		y1 = mean_hits-std_hits
		y1 = np.where(y1>0, y1, 0)
		y2 = mean_hits+std_hits
		y2 = np.where(y2<ideal, y2, ideal)
		ax.fill_between(itr, y1, y2, where=(y2 >= y1), facecolor=color, alpha=0.2, interpolate=True)
		ax.plot(itr, y1, color=color, linewidth=1)
		ax.plot(itr, y2, color=colors, linewidth=1)
	ax.plot(itr, mean_hits, label='ActiveSearch', color=color, linewidth=5)

	# ideal and random chance
	ax.plot(itr, ideal, 'k', label='Ideal', linewidth=4)
	ax.plot(itr, chance, 'ro', label='Chance', alpha=0.5, linewidth=4)

	plt.xlabel('Iterations',fontsize=50)
	plt.ylabel('Number of Hits',fontsize=50)
	plt.legend(loc=1)

	if title:
		plt.title(title, y=1.02, fontsize=50)

	if save:
		fname = title+'.png'
		fig = plt.figure(1)
		fig.set_size_inches(24,14)
		plt.savefig(fname, format='png', transparent=True, facecolor='w')

	# if show:
	plt.show()


def test_ads_single ():

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 300

	# Stratified sampling
	strat_frac = 1.0
	t1 = time.time()
	X,Y,classes = load_training_ad_data(sparse=sparse)
	# Remove 0's
	nz_inds = np.array(X.sum(0)).squeeze().nonzero()[0]
	X = X[:,nz_inds]
	Y = Y[nz_inds]
	print ('Time taken to load: %.2fs'%(time.time()-t1))
	if 0.0 < strat_frac and strat_frac < 1.0:
		t1 = time.time()
		X, Y = du.stratified_sample(X, Y, classes, strat_frac=strat_frac)
		print ('Time taken to sample: %.2fs'%(time.time()-t1))

	# Changing prevalence of +
	prev = 0.05
	t1 = time.time()
	X,Y = du.change_prev (X,Y,prev=prev)
	print ('Time taken change prevalence: %.2fs'%(time.time()-t1))
	X = tfidf_from_word_count(X)
	d,n = X.shape

	# Run Active Search
	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta)
	kAS = ASI.kernelAS (prms)

	num_init = 1
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),num_init,replace=False)]
	kAS.initialize(X, init_labels={p:1 for p in init_pt})

	hits = [len(init_pt)]

	for i in xrange(K):

		idx = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx])

		hits.append(hits[-1]+Y[idx])
	
	IPython.embed()


def test_ads (num_exp=20):

	verbose = False
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 300

	strat_frac = 1.0
	prev = 0.05
	num_init = 1

	t1 = time.time()
	X,Y,classes = load_training_ad_data(sparse=sparse)
	print ('Time taken to load: %.2fs'%(time.time()-t1))
	## Remove 0's
	nz_inds = np.array(X.sum(0)).squeeze().nonzero()[0]
	X = X[:,nz_inds]
	Y = Y[nz_inds]
	# Perform TFIDF
	X = tfidf_from_word_count(X)

	hits_exp = []
	max_hits = []
	for exp in range(num_exp):
		print('Experiment %i out of %i.'%(exp+1, num_exp))

		Xe = X.copy()
		Ye = Y.copy()
		# Stratified sampling
		if 0.0 < strat_frac and strat_frac < 1.0:
			t1 = time.time()
			Xe, Ye = du.stratified_sample(Xe, Ye, classes, strat_frac=strat_frac)
			if verbose:
				print ('Time taken to sample: %.2fs'%(time.time()-t1))

		# Changing prevalence of +
		t1 = time.time()
		Xe,Ye = du.change_prev (Xe,Ye, prev=prev)
		if verbose:
			print ('Time taken change prevalence: %.2fs'%(time.time()-t1))
		d,n = Xe.shape

		# Run Active Search
		prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=verbose, eta=eta)
		kAS = ASI.kernelAS (prms)

		init_pt = Ye.nonzero()[0][nr.choice(len(Ye.nonzero()[0]),num_init,replace=False)]
		kAS.initialize(Xe, init_labels={p:1 for p in init_pt})

		hits = [len(init_pt)]

		for i in xrange(K):

			idx = kAS.getNextMessage()
			kAS.setLabelCurrent(Ye[idx])

			hits.append(hits[-1]+Ye[idx])
		
		hits_exp.append(hits)
		max_hits.append(Ye.sum())

	hits_exp = np.array(hits_exp)
	max_hits = np.max(max_hits)

	plot_expts(hits_exp, max_possible=max_hits, prev=prev)
	IPython.embed()

if __name__ == '__main__':
	matplotlib.rcParams.update({'font.size': 25})
	# test_ads_single()
	test_ads()
	# IPython.embed()