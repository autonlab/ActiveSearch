from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio
import scipy.sparse as ss, scipy.sparse.linalg as ssl
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import time

import os, os.path as osp

import activeSearch as AS
import activeSearchInterface as ASI
from eigenmap import eigenmap
import visualize as vis
import email_features as ef
import gaussianRandomFeatures as grf

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))

def save_sparse_csr(filename,array):
	np.savez(filename,data = array.data ,indices=array.indices,
			 indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
	loader = np.load(filename)
	return ss.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
						 shape = loader['shape'])

def test_vals ():
	import cPickle as pickle
	import os, os.path as osp

	with open(osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/ben/forumthread_SparseMatrix.pkl'),'r') as fl:
		X = pickle.load(fl)
		X = X.T

	# import IPython 
	# IPython.embed()
	n = 100
	r = 100
	X = X[:,:n]
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	X = X[:r,:]
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	X = ss.csr_matrix(np.c_[X.todense(),X[:,50].todense()])
	print X.shape
	r,n = X.shape

	nt = int(0.05*n)
	num_eval = 50
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)
	nr.shuffle(Y)

	pi = sum(Y)/len(Y)
	init_pt = 50


	# A = np.array((X.T.dot(X)).todense())	
	t1 = time.time()

	verbose = True
	w0 = 0.0
	prms = ASI.Parameters(pi=pi, w0=w0, sparse=True, verbose=verbose)
	kAS = ASI.kernelAS(prms)
	kAS.initialize(X)

	import IPython 
	IPython.embed()

	init_lbls = {init_pt:1}

	kAS.firstMessage(init_pt)
	# fs2 = [kAS.f]

	import IPython
	IPython.embed()

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		[idx1]
		import IPython
		IPython.embed()


def test_wikipedia_dataset ():

	fname = osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/sherry_datasets/wikipedia_data.mat')

	data = sio.loadmat(fname)

	threshold = 0.4
	A = data['A']
	labels = data['topic_vectors'][:,105] > threshold
	evals, evecs = ssl.eigsh(A)#,k=A.shape[0])

	num_eval = 600

	pi = sum(labels)/len(labels)
	init_pt = labels.nonzero()[0][nr.randint(len(labels.nonzero()[0]))]

	verbose = True
	prms = ASI.Parameters(pi=pi,sparse=False, verbose=verbose)
	
	sAS = ASI.shariAS(prms)
	sAS.initialize(A.todense().A)

	sAS.firstMessage(init_pt)

	# import IPython
	# IPython.embed()

	for i in range(num_eval):
		idx1 = sAS.getNextMessage()
		sAS.setLabelCurrent(int(labels[idx1]))
		# import IPython
		# IPython.embed()
	import IPython
	IPython.embed()

	tt = sum(labels)
	tremaining = [tt-i for i in sAS.hits]
	plt.plot(tremaining)
	plt.show()

	import IPython
	IPython.embed()

def test_20ng ():
	from sklearn.datasets import fetch_20newsgroups
	from sklearn.feature_extraction.text import CountVectorizer
	# newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
	fng = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'),categories=['alt.atheism','comp.graphics'])
	vectorizer = CountVectorizer( stop_words='english', ngram_range=(1, 1), analyzer=u'word', max_df=0.5, min_df=0.01, binary=True)

	X = vectorizer.fit_transform(fng.data).T
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	
	pfrac = 0.1
	labels = fng.target[np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	nneg = sum(labels==0)
	npos = int(pfrac*nneg/(1-pfrac))

	ninds = (labels==0).nonzero()[0]
	pinds = labels.nonzero()[0]
	nr.shuffle(pinds)
	pinds = pinds[:npos]

	inds = np.r_[pinds,ninds]
	nr.shuffle(inds)

	Xf = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	Xf = Xf[:,inds]
	labels = labels[inds]

	# import IPython
	# IPython.embed()
	
	A = Xf.T.dot(Xf)

	num_eval = 500

	pi = sum(labels)/len(labels)
	init_pt = labels.nonzero()[0][nr.randint(len(labels.nonzero()[0]))]

	verbose = True
	remove_self_degree = False
	prms1 = ASI.Parameters(pi=pi,sparse=True, verbose=verbose, remove_self_degree=remove_self_degree)
	prms2 = ASI.Parameters(pi=pi,sparse=False, verbose=verbose, remove_self_degree=remove_self_degree)

	kAS = ASI.kernelAS(prms1)
	kAS.initialize(Xf)

	sAS = ASI.shariAS(prms2)
	sAS.initialize(A.todense().A)

	kAS.firstMessage(init_pt)
	sAS.firstMessage(init_pt)

	# import IPython
	# IPython.embed()

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(int(labels[idx1]))
		idx2 = sAS.getNextMessage()
		sAS.setLabelCurrent(int(labels[idx2]))
		# import IPython
		# IPython.embed()

	fpr1, tpr1, thresh1 = roc_curve(labels, kAS.f)
	fpr2, tpr2, thresh2 = roc_curve(labels, sAS.f)

	plt.plot(fpr1, tpr1, label='kernel', color='r')
	plt.plot(fpr2, tpr2, label='sherry', color='b')

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic for kernel AS')
	plt.legend(loc="lower right")

	import IPython
	IPython.embed()


def test_20ng_2 ():
	from sklearn.datasets import fetch_20newsgroups
	from sklearn.feature_extraction.text import CountVectorizer
	# newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
	fng = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'),categories=['alt.atheism','comp.graphics'])
	vectorizer = CountVectorizer( stop_words='english', ngram_range=(1, 1), analyzer=u'word', max_df=0.5, min_df=0.01, binary=True)


	X = vectorizer.fit_transform(fng.data).T
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]

	pfrac = 0.1
	labels = fng.target[np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	nneg = sum(labels==0)
	npos = int(pfrac*nneg/(1-pfrac))

	ninds = (labels==0).nonzero()[0]
	pinds = labels.nonzero()[0]
	nr.shuffle(pinds)
	pinds = pinds[:npos]

	inds = np.r_[pinds,ninds]
	nr.shuffle(inds)

	Xf = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	Xf = Xf[:,inds]
	labels = labels[inds]

	import IPython
	IPython.embed()


	A = Xf.T.dot(Xf)

	num_eval = 500

	pi = sum(labels)/len(labels)
	init_pt = labels.nonzero()[0][nr.randint(len(labels.nonzero()[0]))]

	verbose = True
	rsd1 = True
	rsd2 = False
	prms1 = ASI.Parameters(pi=pi,sparse=False, verbose=verbose, remove_self_degree=rsd1)
	prms2 = ASI.Parameters(pi=pi,sparse=False, verbose=verbose, remove_self_degree=rsd2)

	sAS1 = ASI.shariAS(prms1)
	sAS1.initialize(A.todense().A)
	sAS2 = ASI.shariAS(prms2)
	sAS2.initialize(A.todense().A)

	sAS1.firstMessage(init_pt)
	sAS2.firstMessage(init_pt)

	# import IPython
	# IPython.embed()

	for i in range(num_eval):
		idx1 = sAS1.getNextMessage()
		sAS1.setLabelCurrent(int(labels[idx1]))
		idx2 = sAS2.getNextMessage()
		sAS2.setLabelCurrent(int(labels[idx2]))
		# import IPython
		# IPython.embed()


	tt = sum(labels)
	tremaining = [tt-i for i in sAS1.hits]
	tremaining2 = [tt-i for i in sAS2.hits]
	plt.plot(tremaining)
	plt.plot(tremaining2,color='r')
	plt.show()

	import IPython
	IPython.embed()


def test_logistic_reg ():
	from sklearn.datasets import fetch_20newsgroups
	from sklearn.feature_extraction.text import CountVectorizer


	# newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
	fng = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'),categories=['alt.atheism','comp.graphics'])
	vectorizer = CountVectorizer( stop_words='english', ngram_range=(1, 1), analyzer=u'word', max_df=0.5, min_df=0.01, binary=True)

	X = vectorizer.fit_transform(fng.data).T
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	labels = fng.target[np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	pfrac = 0.1
	labels = fng.target[np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	nneg = sum(labels==0)
	npos = int(pfrac*nneg/(1-pfrac))

	ninds = (labels==0).nonzero()[0]
	pinds = labels.nonzero()[0]
	nr.shuffle(pinds)
	pinds = pinds[:npos]

	inds = np.r_[pinds,ninds]
	nr.shuffle(inds)

	Xf = matrix_squeeze(X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))].todense())
	Xf = Xf[:,inds]
	labels = labels[inds]
	
	# import IPython
	# IPython.embed()


	num_eval = 100
	all_fs = True
	C = .03

	pi = sum(labels)/len(labels)
	init_pt = labels.nonzero()[0][nr.randint(len(labels.nonzero()[0]))]

	verbose = True
	remove_self_degree = False

	alpha = 0.5
	sfunc1='PointWiseMultiply'
	sfunc2='AbsDifference'
	sfunc3='LinearKernel'

	# f1, _, _, _, _ =	AS.regression_active_search (Xf, labels, num_eval=num_eval, w0=None, pi=pi, eta=0.5, alpha=alpha, C=C, sfunc=sfunc1, init_pt=init_pt, verbose=verbose, all_fs=all_fs)
	f2, _, _, _, _ =	AS.regression_active_search (Xf, labels, num_eval=num_eval, w0=None, pi=pi, eta=0.5, alpha=alpha, C=C, sfunc=sfunc2, init_pt=init_pt, verbose=verbose, all_fs=all_fs)
	# f3, _, _, _, _ =	AS.regression_active_search (Xf, labels, num_eval=num_eval, w0=None, pi=pi, eta=0.5, alpha=alpha, C=C, sfunc=sfunc3, init_pt=init_pt, verbose=verbose, all_fs=all_fs)

	# q1 = np.percentile(f1,pi*100)
	q2 = np.percentile(f2,pi*100)
	# q3 = np.percentile(f3,pi*100)
	# found_l1 = (f1>q1).astype(int)
	found_l2 = (f2>q2).astype(int)
	# found_l3 = (f3>q3).astype(int)
	# v = sum([i==j for i,j in zip(found_l,labels)])/Xf.shape[1]

	# fpr1, tpr1, thresh1 = roc_curve(labels, f1)
	fpr2, tpr2, thresh2 = roc_curve(labels, f2)
	# fpr3, tpr3, thresh3 = roc_curve(labels, f3)

	# plt.plot(fpr1, tpr1, label='AbsDiff', color='r')
	plt.plot(fpr2, tpr2, label='Ptwise', color='b')
	# plt.plot(fpr3, tpr3, label='Linear', color='g')

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic, alpha = %f'%alpha)
	plt.legend(loc="lower right")

	import IPython
	IPython.embed()


def test_20_ng_IM ():
	from sklearn.datasets import fetch_20newsgroups
	from sklearn.feature_extraction.text import CountVectorizer


	# newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
	fng = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'),categories=['alt.atheism','comp.graphics'])
	vectorizer = CountVectorizer( stop_words='english', ngram_range=(1, 1), analyzer=u'word', max_df=0.5, min_df=0.01, binary=True)

	X = vectorizer.fit_transform(fng.data).T
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	labels = fng.target[np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	pfrac = 0.1
	labels = fng.target[np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	nneg = sum(labels==0)
	npos = int(pfrac*nneg/(1-pfrac))

	ninds = (labels==0).nonzero()[0]
	pinds = labels.nonzero()[0]
	nr.shuffle(pinds)
	pinds = pinds[:npos]

	inds = np.r_[pinds,ninds]
	nr.shuffle(inds)

	# Xf = matrix_squeeze(X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))].todense())
	# Xf = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	Xf = X[:,inds]
	labels = labels[inds]
	
	# import IPython
	# IPython.embed()


	num_eval = 100
	all_fs = True
	C = .03

	pi = sum(labels)/len(labels)
	init_pt = labels.nonzero()[0][nr.randint(len(labels.nonzero()[0]))]

	verbose = True
	remove_self_degree = False
	sparse = True
	alpha1 = 0.00
	alpha2 = 0.1
	alpha3 = 0.2
	alpha4 = 0.5


	f1, _, _, _ =	AS.kernel_AS (Xf, labels, num_eval=num_eval, w0=None, pi=pi, eta=0.5, alpha=alpha1, init_pt=init_pt, sparse=sparse, verbose=verbose, all_fs=all_fs)
	f2, _, _, _ =	AS.kernel_AS (Xf, labels, num_eval=num_eval, w0=None, pi=pi, eta=0.5, alpha=alpha2, init_pt=init_pt, sparse=sparse, verbose=verbose, all_fs=all_fs)
	f3, _, _, _ =	AS.kernel_AS (Xf, labels, num_eval=num_eval, w0=None, pi=pi, eta=0.5, alpha=alpha3, init_pt=init_pt, sparse=sparse, verbose=verbose, all_fs=all_fs)
	f4, _, _, _ =	AS.kernel_AS (Xf, labels, num_eval=num_eval, w0=None, pi=pi, eta=0.5, alpha=alpha4, init_pt=init_pt, sparse=sparse, verbose=verbose, all_fs=all_fs)

	# q1 = np.percentile(f1,pi*100)
	# found_l1 = (f1>q1).astype(int)

	fpr1, tpr1, thresh1 = roc_curve(labels, f1)
	fpr2, tpr2, thresh2 = roc_curve(labels, f2)
	fpr3, tpr3, thresh3 = roc_curve(labels, f3)
	fpr4, tpr4, thresh4 = roc_curve(labels, f4)

	plt.plot(fpr1, tpr1, label='alpha=%f'%alpha1, color='r')
	plt.plot(fpr2, tpr2, label='alpha=%f'%alpha2, color='g')
	plt.plot(fpr3, tpr3, label='alpha=%f'%alpha3, color='b')
	plt.plot(fpr4, tpr4, label='alpha=%f'%alpha4, color='k')


	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')#, alpha = %f'%alpha)
	plt.legend(loc="lower right")

	import IPython
	IPython.embed()


if __name__ == '__main__':

	# test_vals()
	# test_wikipedia_dataset()
	# test_20ng ()
	#test_20ng_2 ()
	# test_logistic_reg ()
		test_20_ng_IM ()