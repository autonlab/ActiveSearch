from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
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

def save_sparse_csr(filename,array):
	np.savez(filename,data = array.data ,indices=array.indices,
			 indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
	loader = np.load(filename)
	return ss.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
						 shape = loader['shape'])

def test1 (n, cc=2, nt=1, d=4):
	# Example where you chains of connections in targets
	# and similar chains in non-targets

	num_eval = 5
	init_pt = 1
	# Created banded diagonal kernel matrix
	X = np.eye(n)
	X[range(n-1), range(1,n)] = 1
	Xfull = slg.block_diag(*([X]*cc))
	Yfull = [1]*(nt*n) + [0]*((cc-nt)*n)

	pi = sum(Yfull)/len(Yfull)

	f1,_,_ = AS.kernel_AS (Xfull, Yfull, pi=pi, num_eval=num_eval, init_pt=init_pt)

	Xe, b, w, deg = eigenmap(Xfull.T.dot(Xfull), d)
	f2,_,_ = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Yfull, options={'num_eval':num_eval,'pi':pi,'init_pt':init_pt,'n_conncomp':b})

	import IPython
	IPython.embed()



def test2 (n, cc=2, nt=1, d=5):

	# Created banded diagonal kernel matrix
	X = np.eye(n)
	X[range(n-1), range(1,n)] = 1
	Xfull = slg.block_diag(*([X]*cc))
	Yfull = [0]+[1]+[0]*(n-5)+[1]*3+[1]*((nt-1)*n) + [0]*((cc-nt)*n)

	Xe, b, w, deg = eigenmap(Xfull.T.dot(Xfull), d)

	num_eval = 5
	init_pt = 1
	pi = sum(Yfull)/len(Yfull)

	f1 = AS.kernel_AS (Xfull, Yfull, pi=pi, num_eval=num_eval, init_pt=init_pt)
	f2 = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Yfull, options={'num_eval':num_eval,'pi':pi,'init_pt':init_pt,'n_conncomp':b})

	import IPython
	IPython.embed()


def test3 (hubs=3, followers=10):

	# Creating as follows:
	# Every alternate hub and half its followers are targets.
	# Every hub shares followers with adjacent hubs

	fl0 = int(followers/2)
	fl1 = followers - fl0

	X = np.zeros((hubs*followers, (hubs-1)*followers + hubs))
	Yfull = []
	t = 0
	for i in range(hubs-1):
		idx = i*(followers+1)
		X[i*followers:(i+1)*followers, idx] = 1
		X[xrange(i*followers,(i+1)*followers), xrange(idx+1, idx+followers+1)] = [1]*followers
		X[xrange((i+1)*followers,(i+2)*followers), xrange(idx+1, idx+followers+1)] = [1]*followers
		if t == 0:
			Yfull += [0]*(fl0+1) + [1]*fl1
			t = 1
		else:
			Yfull += [1]*(fl1+1) + [0]*fl0
			t = 0

	X[(hubs-1)*followers:, -1] = 1
	if t == 0: 
		Yfull.append(0)
	else:
		Yfull.append(1)

	Xfull = X

	d = 10


	num_eval = 15
	init_pt = 10
	pi = sum(Yfull)/len(Yfull)

	# t1 = time.time()
	f1,_,_ = AS.kernel_AS (Xfull, Yfull, pi=pi, num_eval=num_eval, init_pt=init_pt)
	# t2 = time.time()
	Xe, b, w, deg = eigenmap(Xfull.T.dot(Xfull), d)
	f2,_,_ = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Yfull, options={'num_eval':num_eval,'pi':pi,'init_pt':init_pt,'n_conncomp':b})
	# t3 = time.time()

	# print "Time 1:", (t2-t1)
	# print "Time 2:", (t3-t2)


	# print f1
	# print f2

	plt.plot(f1, color='r', label='kernel')
	plt.plot(f2, color='b', label='lreg')
	plt.plot(Yfull, color='g', label='true')
	plt.legend()
	plt.show()

	mat = sio.loadmat('../matlab/L.mat')
	Xe2 = mat['Xe']
	deg2 = np.squeeze(mat['deg'])
	Yfull2 = np.squeeze(mat['Yfull'])
	f3,_,_ = AS.lreg_AS (Xe2, deg2, d, alpha=0.0, labels=Yfull2, options={'num_eval':num_eval,'pi':pi,'init_pt':init_pt,'n_conncomp':b})

	import IPython
	IPython.embed()

def createFakeData (n, r, nt, rcross=10):
	"""
	Builds data set
	"""

	low, high = 0, 5/n
	low_c, high_c = 0, 1/(n**2)

	Xt_t = nr.uniform(low=low, high=high, size=(r,nt))
	Xt_n = nr.uniform(low=low_c, high=high_c, size=(r - rcross, nt))
	Xn_n = nr.uniform(low=low, high=high, size=(r,n-nt))
	Xn_t = nr.uniform(low=low_c, high=high_c, size=(r - rcross, n-nt))

	X = np.c_[np.r_[Xt_t, Xt_n],np.r_[Xn_t, Xn_n]]
	Y = np.array([1]*nt + [0]*(n-nt))

	return X, Y


def createFakeData2 (n, r, nt, hubs):
	"""
	Builds data set
	"""

	low, high = 0, 5/n

	X = nr.uniform(low=low, high=high, size=(r,n))
	hub_vecs = nr.uniform(low=low, high=high, size=(r,hubs))

	hubX = X.T.dot(hub_vecs)

	sortedX = np.sort(hubX, axis=0)
	if hubs == 3:
		dist = sum(sortedX[int(nt/2.7), :])/3
	elif hubs == 1:
		dist = sum(sortedX[int(nt), :])

	Y = (np.sum(hubX < dist, axis=1) > 0).astype(int)

	# import IPython
	# IPython.embed()

	return X, Y

def createFakeData3 (n, r, nt, rcross=10):
	"""
	Builds data set
	"""

	low, high = 0, 5/n

	Xt_t = nr.uniform(low=low, high=high, size=(r,nt))
	Xt_n = np.zeros((r - rcross, nt))
	Xn_n = nr.uniform(low=low, high=high, size=(r,n-nt))
	Xn_t = np.zeros((r - rcross, n-nt))

	X = np.c_[np.r_[Xt_t, Xt_n],np.r_[Xn_t, Xn_n]]
	Y = np.array([1]*nt + [0]*(n-nt))

	return X, Y


def compute_f (A, labels, selected, l, w0, pi):
	"""
	Immediately compute f given selected points.
	"""
	n = A.shape[0]

	B = np.ones(n)/(1+w0)
	B[selected] = l/(l+1)
	D = np.squeeze(A.dot(np.ones((n,1))))
	BDinv = B*(1./D)

	Ap = np.diag(BDinv).dot(A)

	y = np.ones(n)*pi
	y[selected] = labels[selected]
	q = np.diag(1-B).dot(y)

	f = nlg.inv(np.eye(n)-Ap).dot(q)

	return f


def test4():

	n = 1000
	r = 100
	nt = 100
	#rcross = 50
	d = 10
	hubs = 1

	verbose = False

	num_eval = 300
	#init_pt = 1

	#X, Y = createFakeData(n, r, nt, rcross)
	X, Y = createFakeData2(n, r, nt, hubs)

	pi = sum(Y)/len(Y)
	print "Constructing the similarity matrix:"
	A = X.T.dot(X)
	t1 = time.time()
	print "Performing Kernel AS"

	f1,h1,s1 = AS.kernel_AS (X, Y, pi=pi, num_eval=num_eval, init_pt=None, verbose=verbose)
	t2 = time.time()
	print "Performing Eigen decmop"
	Xe, b, w, deg = eigenmap(A, d)
	t3 = time.time()
	print "Performing LREG AS"
	f2,h2,s2 = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Y, options={'num_eval':num_eval,'pi':pi,'n_conncomp':b}, verbose=verbose)
	t4 = time.time()

	print "Time taken for kernel:", t2-t1
	print "Time taken for eigenmap + computing X.T*X:", t3-t2
	print "Time taken for lreg:", t4-t3
	print "h_kernel: %i/%i"%(h1[-1],num_eval)
	print "h_lreg: %i/%i"%(h2[-1],num_eval)

	import IPython
	IPython.embed()

def testfake():

	n = 400
	r = 60
	nt = 150
	#rcross = 50
	d = n
	hubs = 1

	verbose = False

	num_eval = 390
	#init_pt = 1

	for i in range(100):
		#X, Y = createFakeData(n, r, nt, rcross)
		X, Y = createFakeData2(n, r, nt, hubs)

		pi = sum(Y)/len(Y)
		print i

		# print "Constructing the similarity matrix:"
		# print "Performing Kernel AS"
		f1,h1,s1 = AS.kernel_AS (X, Y, pi=pi, num_eval=num_eval, init_pt=None, verbose=verbose)


def test5():

	verbose = False
	mat = sio.loadmat('../matlab/M.mat')
	n = int(mat['n'])
	r = int(mat['r'])
	nt = int(mat['nt'])
	rcross = int(mat['rcross'])
	d = int(mat['d'])
	num_eval = mat['num_eval']
	init_pt = int(mat['init_pt'])
	Xfull = mat['Xfull']
	
	deg = np.squeeze(mat['deg'])
	Yfull = np.squeeze(mat['Yfull'])
	pi = sum(Yfull)/len(Yfull)
	
	t1 = time.time()
	Xe, b, w, deg = eigenmap(Xfull.T.dot(Xfull), d)
	t2 = time.time()
	f2,_,_ = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Yfull, options={'num_eval':num_eval,'pi':pi,'n_conncomp':b,'init_pt':init_pt}, verbose=verbose)
	t3 = time.time()
	f1,_,_ = AS.kernel_AS (Xfull, Yfull, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose)
	t4 = time.time()

	print "Time taken for eigenmap + computing X.T*X:", t2-t1
	print "Time taken for lreg:", t3-t2
	print "Time taken for kernel:", t4-t3

	f1 = np.squeeze(f1)
	f2 = np.squeeze(f2)
	f3 = np.squeeze(mat['f'])

	import IPython
	IPython.embed()


# def run_kernel_AS()

def test6():
	n = 10
	start = 1000
	end = 5000
	nt_ratio = 0.6
	r_ratio = 0.1
	ne_ratio = 0.2

	nrange = np.logspace(np.log10(start),np.log10(end), n).tolist()
	nrange = [int(nv) for nv in nrange]
	ntrange = [int(nt_ratio*nv) for nv in nrange]
	rrange = [int(r_ratio*nv) for nv in nrange]
	drange = [r*2 for r in rrange]
	nerange = [int(ne_ratio*nv) for nv in nrange]

	t_kernel = []
	t_eigendecomp = []
	t_lreg = []
	max_error = []

	verbose = False
	for n,nt,r,d,ne in zip(nrange,ntrange,rrange,drange,nerange):

		X, Y = createFakeData(n, r, nt, rcross=0)

		pi = sum(Y)/len(Y)
	
		A = X.T.dot(X) # Don't really want to include timings for this.

		t1 = time.time()
		f1,h1,s1 = AS.kernel_AS (X, Y, pi=pi, num_eval=ne, init_pt=None, verbose=verbose)
		t2 = time.time()
		Xe, b, w, deg = eigenmap(A, d)
		t3 = time.time()
		f2,h2,s2 = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Y, options={'num_eval':ne,'pi':pi,'n_conncomp':b}, verbose=verbose)
		t4 = time.time()

		print "Parameters: n=%i, nt=%i, r=%i, d=%i, ne=%i"%(n,nt,2*r,d,ne)
		print "Time taken for kernel:", t2-t1
		print "Time taken for eigenmap + computing X.T*X:", t3-t2
		print "Time taken for lreg:", t4-t3
		print "Max error difference:", np.max(np.abs(f1-f2))
		print

		t_kernel.append(t2-t1)
		t_eigendecomp.append(t3-t2)
		t_lreg.append(t4-t3)
		max_error.append(np.max(np.abs(f1-f2)))

	plt.plot(nrange, t_kernel, label='kernel')
	plt.plot(nrange, t_eigendecomp, label='eigendecomp')
	plt.plot(nrange, t_lreg, label='lreg')

	plt.legend()
	plt.xlabel('Number of data points')
	plt.ylabel('Time')

	plt.show()

	import IPython
	IPython.embed()



def test8():

	n = 500
	r = 50
	nt = 100
	#rcross = 50
	d = 50
	hubs = 1

	verbose = False

	num_eval = 100

	#X, Y = createFakeData(n, r, nt, rcross)
	#X, Y = createFakeData2(n, r, nt, hubs)
	X,Y = np.load('t8data.npy')
	# import IPython
	# IPython.embed()

	init_pt = np.nonzero(Y)[0][0]

	ker = True


	pi = sum(Y)/len(Y)
	print "Constructing the similarity matrix:"
	A = X.T.dot(X)
	t1 = time.time()
	if ker:
		print "Performing Kernel AS"
		f1,h1,s1,fs1 = AS.kernel_AS (X, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True)
	t2 = time.time()
	#print "Performing Eigen decmop"
	#Xe, b, w, deg = eigenmap(A, d)
	#t3 = time.time()
	if ker:
		print "Performing Naive Shari AS"
		f2,h2,s2,fs2 = AS.shari_activesearch_probs_naive(A, labels=Y, pi=pi, w0=None, eta=None, num_eval=num_eval, init_pt=init_pt, verbose=verbose, all_fs=True)
		#f2,h2,s2,fs2 = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Y, options={'num_eval':num_eval,'pi':pi,'n_conncomp':b}, verbose=verbose)
	t4 = time.time()

	print "Time taken for kernel:", t2-t1
	#print "Time taken for eigenmap + computing X.T*X:", t3-t2
	print "Time taken for Shari's method (naive):", t4-t2
	if ker:
		print "h_kernel: %i/%i"%(h1[-1],num_eval)
		print "h_lreg: %i/%i"%(h2[-1],num_eval)

	import IPython
	IPython.embed()


def test9():
	verbose = True

	datadir = osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/sibi_matrices')
	tsfile = osp.join(datadir, 'timestamps.csv')
	tffile = osp.join(datadir, 'tfidf_pretranspose.txt')
	contactsfile = osp.join(datadir, 'email_person_bitarray.txt')

	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')

	n = 5000
	r = 2000
	nt = 15#int(0.1*n)
	num_eval = nt*2
	# getting rid of features which are zero for all these elements
	X = np.array((Xfull[:,:n]).todense())
	X = X[np.nonzero(X.sum(axis=1))[0],:]
	X = X[:,np.nonzero(X.sum(axis=0))[1]]
	# import IPython 
	# IPython.embed()
	X = X[:r,:]
	X = X[np.nonzero(X.sum(axis=1))[0],:]
	X = X[:,np.nonzero(X.sum(axis=0))[1]]
	# import IPython 
	# IPython.embed()

	r,n = X.shape
	d = 20
	nt = 10#int(0.1*n)
	num_eval = 15#nt*2
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)

	pi = nt*1.0/n
	init_pt = 100

	A = X.T.dot(X)
	import IPython
	IPython.embed()

	t1 = time.time()
	print "Kernel method"
	#f1,h1,s1,fs1,dt = AS.kernel_AS (X, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True,tinv=True)
	t2 = time.time()
	print "Eigen map"
	#Xe, b, w, deg = eigenmap(A, d)
	#np.save('eigenstuff',[Xe, b, w, deg])
	Xe, b, w, deg  = np.load('eigenstuff.npy')
	# import IPython 
	# IPython.embed()
	t3 = time.time()
	print "Shari method"
	#f2,h2,s2 = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Y, options={'num_eval':num_eval,'pi':pi,'n_conncomp':b,'init_pt':init_pt}, verbose=verbose)
	t4 = time.time()
	f3,h3,s3,fs3 = AS.shari_activesearch_probs_naive(A, labels=Y, pi=pi, w0=None, eta=None, num_eval=num_eval, init_pt=init_pt, verbose=verbose, all_fs=True)

	print "Time taken for kernel:", t2-t1
	#print "Time taken for inverse:", dt
	print "Time taken for eigen decomp:", t3 - t2
	print "Time taken for lreg:", t4-t3


	#f1 = np.squeeze(f1)

	import IPython
	IPython.embed()


def test10():
	verbose = False
	Xe, b, w, deg  = np.load('eigenstuff.npy')
	n = Xe.shape[0]

	nt = int(0.1*n)
	num_eval = nt*2
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)

	drange = range(10) + [10*i for i in range(2,11)] + [100*i for i in range(2,11,2)] + [1500, 2000, 2500]

	pi = sum(Y)/len(Y)
	init_pt = 100

	hits = []
	rtime = []

	for d in drange:
		t1 = time.time()
		f2,h2,s2 = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Y, options={'num_eval':num_eval,'pi':pi,'n_conncomp':b,'init_pt':init_pt}, verbose=verbose)
		t2 = time.time()

		hits.append(h2[-1][0])
		rtime.append(t2-t1)

		print "d =", d
		print "hits =", h2[-1][0]
		print "time =", t2-t1
		print
		print
	#f3,h3,s3,fs3 = AS.shari_activesearch_probs_naive(A, labels=Y, pi=pi, w0=None, eta=None, num_eval=num_eval, init_pt=init_pt, verbose=verbose, all_fs=True)

	# print "Time taken for kernel:", t2-t1
	# #print "Time taken for inverse:", dt
	# print "Time taken for eigen decomp:", t3 - t2
	# print "Time taken for lreg:", t4-t3

	#f1 = np.squeeze(f1)
	# plt.figure()
	# plt.plot(drange, hits)
	# plt.figure()
	# plt.plot(drange, rtime)
	# plt.show()

	# import IPython
	# IPython.embed()

	return drange, hits, rtime

def test11():

	verbose = False
	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')

	n = 5000
	nt = int(0.1*n)
	num_eval = nt*2
	# getting rid of features which are zero for all these elements
	# X = np.array((Xfull[:,:n]).todense())
	# X = X[np.nonzero(X.sum(axis=1))[0],:]
	# X = X[:,np.nonzero(X.sum(axis=0))[1]]
	# import IPython 
	# IPython.embed()
	# X = X[:r,:]
	# X = X[np.nonzero(X.sum(axis=1))[0],:]
	# X = X[:,np.nonzero(X.sum(axis=0))[1]]
	X = np.load('X11.npy')

	# import IPython
	# IPython.embed()

	nt = int(0.1*n)
	num_eval = nt*2
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)

	rrange =  [10*i for i in range(2,11)] + [100*i for i in range(2,11,2)] + [1500, 2000, 2500]

	pi = sum(Y)/len(Y)
	init_pt = 100

	hits = []
	rtime = []

	for r in rrange:
		Xr = X[:r,:]
		Xr = Xr[np.nonzero(Xr.sum(axis=1))[0],:]
		Xr = Xr[:,np.nonzero(Xr.sum(axis=0))[1]]

		t1 = time.time()
		f1,h1,s1,fs1 = AS.kernel_AS (Xr, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True)
		t2 = time.time()

		hits.append(h1[-1][0])
		rtime.append(t2-t1)

		print "r =", r
		print "hits =", h1[-1][0]
		print "time =", t2-t1
		print
		print

	# plt.figure()
	# plt.plot(rrange, hits)
	# plt.xlabel('d')
	# plt.ylabel('hits')
	# plt.figure()
	# plt.plot(rrange, rtime)
	# plt.xlabel('d')
	# plt.ylabel('time in s')
	# plt.show()

	# import IPython
	# IPython.embed()
	return rrange, hits, rtime


def test12():

	verbose = True
	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')

	r,n = Xfull.shape
		
	nt = int(0.1*n)
	num_eval = nt*2
	X = np.array(Xfull.todense())
	# getting rid of features which are zero for all these elements
	X = X[np.nonzero(X.sum(axis=1))[0],:]
	X = X[:,np.nonzero(X.sum(axis=0))[1]]
	# import IPython 
	# IPython.embed()
	X = X[:r,:]
	X = X[np.nonzero(X.sum(axis=1))[0],:]
	X = X[:,np.nonzero(X.sum(axis=0))[1]]
	#X2 = np.load('X11.npy')

	# import IPython
	# IPython.embed()

	nt = int(0.1*n)
	num_eval = nt*2
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)

	#rrange =  [10*i for i in range(2,11)] + [100*i for i in range(2,11,2)] + [1500, 2000, 2500, 3000]

	pi = sum(Y)/len(Y)
	init_pt = 100

	#hits = []
	#rtime = []

	#for r in rrange:
	
	t1 = time.time()
	print "Performing the kernel AS"
	f1,h1,s1,fs1 = AS.kernel_AS (X, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True,sparse=False)
	t2 = time.time()

	# hits.append(h1[-1][0])
	# rtime.append(t2-t1)

	# print "r =", r
	# print "hits =", h1[-1][0]
	print "time =", t2-t1
	# print
	# print

	# plt.figure()
	# plt.plot(rrange, hits)
	# plt.xlabel('d')
	# plt.ylabel('hits')
	# plt.figure()
	# plt.plot(rrange, rtime)
	# plt.xlabel('d')
	# plt.ylabel('time in s')
	# plt.show()

	# Timing
	# Constructing C: 71s
	# Inverse of C: 202s
	# Total time: 3446s
	import IPython
	IPython.embed()

def test13(): # testing sparse AS

	verbose = True
	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')
	print Xfull.shape
	Xfull = Xfull[np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=1))[0])),:]
	Xfull = Xfull[:,np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=0))[1]))]
	r,n = Xfull.shape

	nt = int(0.1*n)
	num_eval = 20
	# getting rid of features which are zero for all these elements
	# X = np.array((Xfull[:,:n]).todense())
	# X = X[np.nonzero(X.sum(axis=1))[0],:]
	# X = X[:,np.nonzero(X.sum(axis=0))[1]]
	# import IPython 
	# IPython.embed()
	# X = X[:r,:]
	# X = X[np.nonzero(X.sum(axis=1))[0],:]
	# X = X[:,np.nonzero(X.sum(axis=0))[1]]
	#X = np.load('X11.npy')

	# import IPython
	# IPython.embed()
	num_eval = 20
	# num_eval = nt*2
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)


	pi = sum(Y)/len(Y)
	init_pt = 100

	t1 = time.time()
	f1,h1,s1,fs1,dtinv1 = AS.kernel_AS (Xfull, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True,tinv=True,sparse=True)
	t2 = time.time()
	f2,h2,s2,fs2,dtinv2 = AS.kernel_AS (np.array(Xfull.todense()), Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True,tinv=True,sparse=False)
	t3 = time.time()

	import IPython
	IPython.embed()

def test_interface ():
	verbose = False
	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')

	r,n = Xfull.shape

	nt = int(0.05*n)
	num_eval = 1000
	# num_eval = nt*2
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)

	pi = sum(Y)/len(Y)
	init_pt = 100

	t1 = time.time()

	prms = ASI.Parameters(pi=pi,sparse=True, verbose=verbose)	
	kAS = ASI.kernelAS(prms)
	kAS.initialize(Xfull)
	kAS.firstMessage(init_pt)
	fs2 = [kAS.f]	

	for i in range(num_eval):
		idx = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx])
		fs2.append(kAS.f)

	t2 = time.time()

	f1,h1,s1,fs1,dtinv1 = AS.kernel_AS (Xfull, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True,tinv=True,sparse=True)

	t3 = time.time()

	checks = [np.allclose(fs1[i],fs2[i]) for i in range(len(fs1))]

	import IPython
	IPython.embed()


def test_interface2 ():
	verbose = True
	nac = np.allclose
	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')
	print Xfull.shape
	# Xfull = Xfull[np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=1))[0])),:]
	# Xfull = Xfull[:,np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=0))[1]))]
	# r,n = Xfull.shape

	# print Xfull.shape
	# Xfull = Xfull[np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=1))[0])),:]
	# Xfull = Xfull[:,np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=0))[1]))]

	# getting rid of features which are zero for all these elements
	n = 300
	r = 600
	X = Xfull[:,:n]

	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	X = X[:r,:]
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	print X.shape
	#X = np.load('X11.npy')
	r,n = X.shape
	
	nt = int(0.05*n)
	num_eval = 50
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)

	pi = sum(Y)/len(Y)
	init_pt = 5

	# import IPython 
	# IPython.embed()

	A = np.array((X.T.dot(X)).todense())
	t1 = time.time()

	prms = ASI.Parameters(pi=pi,sparse=True, verbose=verbose)	
	kAS = ASI.kernelAS(prms)
	kAS.initialize(X)
	sAS = ASI.naiveShariAS(prms)
	sAS.initialize(A)
	
	import IPython
	IPython.embed()

	kAS.firstMessage(init_pt)
	fs2 = [kAS.f]
	sAS.firstMessage(init_pt)
	fs3 = [sAS.f]

	import IPython
	IPython.embed()

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		idx2 = sAS.getNextMessage()
		print('NEXT')
		print idx1==idx2
		print nac(kAS.f, sAS.f)
		# import IPython
		# IPython.embed()

		kAS.setLabelCurrent(Y[idx1])
		sAS.setLabelCurrent(Y[idx2])
		fs2.append(kAS.f)
		fs3.append(sAS.f)

	t2 = time.time()

	# f1,h1,s1,fs1,dtinv1 = AS.kernel_AS (Xfull, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True,tinv=True,sparse=True)

	t3 = time.time()

	# checks = [np.allclose(fs1[i],fs2[i]) for i in range(len(fs1))]

	import IPython
	IPython.embed()


def test_interface3 ():

	verbose = True
	nac = np.allclose
	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')
	print Xfull.shape
	Xfull = Xfull[np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=1))[0])),:]
	Xfull = Xfull[:,np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=0))[1]))]
	# r,n = Xfull.shape

	print Xfull.shape
	Xfull = Xfull[np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=1))[0])),:]
	Xfull = Xfull[:,np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=0))[1]))]

	# getting rid of features which are zero for all these elements
	# n = 300
	# r = 600
	X = Xfull#[:,:n]

	# X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	# X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	# X = X[:r,:]
	# X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	# X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	# print X.shape
	# #X = np.load('X11.npy')
	r,n = X.shape
	
	nt = int(0.05*n)
	num_eval = 50
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)

	pi = sum(Y)/len(Y)
	init_pt = 5

	# import IPython 
	# IPython.embed()

	A = np.array((X.T.dot(X)).todense())
	t1 = time.time()

	prms = ASI.Parameters(pi=pi,sparse=True, verbose=verbose)	
	kAS = ASI.kernelAS(prms)
	kAS.initialize(X)
	sAS = ASI.shariAS(prms)
	sAS.initialize(A)

	# ofk = kAS.f
	# ofs = sAS.f

	# import IPython
	# IPython.embed()


	kAS.firstMessage(init_pt)
	fs2 = [kAS.f]
	sAS.firstMessage(init_pt)
	fs3 = [sAS.f]

	# #
	# lbl = 1
	# idx = 5
	
	# B = np.ones(n)/(1+prms.w0)
	# D = A.sum(axis=1)
	# BDinv = np.diag(np.squeeze(B*1./D))
	# IA = np.eye(n) - BDinv.dot(A)
	# IAi = np.matrix(nlg.inv(IA))
	# IAk = nlg.inv(np.eye(n) + kAS.BDinv.dot(X.T.dot(nlg.inv(np.eye(r) - X.dot(kAS.BDinv.dot(X.T))))).dot(X.todense()))
	# IAki = nlg.inv(IAk)
	
	# t = (1+prms.w0)*(1-prms.eta)
	# e = np.zeros((n,1))
	# e[idx] = 1
	# IA2 = IA + (1-t)*e.dot(e.T).dot(BDinv.dot(A))
	# ai = (1./D)[idx]/(1+ prms.w0)*A[idx,:]
	# Ad = (1-t)*IAi[:,idx].dot(ai.dot(IAi))/(1 + (1-t)*ai.dot(IAi[:,idx]))
	# IA2i = IAi - Ad
	#


	# import IPython
	# IPython.embed()

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		idx2 = sAS.getNextMessage()
		print('NEXT')
		print idx1==idx2
		print nac(kAS.f, sAS.f)
		# import IPython
		# IPython.embed()

		kAS.setLabelCurrent(Y[idx1])
		sAS.setLabelCurrent(Y[idx2])
		fs2.append(kAS.f)
		fs3.append(sAS.f)

	t2 = time.time()

	# f1,h1,s1,fs1,dtinv1 = AS.kernel_AS (Xfull, Y, pi=pi, num_eval=num_eval, init_pt=init_pt, verbose=verbose,all_fs=True,tinv=True,sparse=True)

	t3 = time.time()

	# checks = [np.allclose(fs1[i],fs2[i]) for i in range(len(fs1))]

	import IPython
	IPython.embed()


def test_warm_start ():

	verbose = True
	nac = np.allclose
	#ts_data = ef.load_timestamps (tsfile)
	Xfull = load_sparse_csr('Xfull1.npz')
	# print Xfull.shape
	# Xfull = Xfull[np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=1))[0])),:]
	# Xfull = Xfull[:,np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=0))[1]))]
	# # r,n = Xfull.shape

	# print Xfull.shape
	# Xfull = Xfull[np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=1))[0])),:]
	# Xfull = Xfull[:,np.squeeze(np.asarray(np.nonzero(Xfull.sum(axis=0))[1]))]

	# getting rid of features which are zero for all these elements
	n = 300
	r = 600
	X = Xfull[:,:n]

	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	X = X[:r,:]
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	print X.shape
	#X = np.load('X11.npy')
	r,n = X.shape
	
	nt = int(0.05*n)
	num_eval = 50
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)
	nr.shuffle(Y)

	pi = sum(Y)/len(Y)
	init_pt = 5

	# import IPython 
	# IPython.embed()

	A = np.array((X.T.dot(X)).todense())
	t1 = time.time()

	prms = ASI.Parameters(pi=pi,sparse=True, verbose=verbose)	
	kAS = ASI.kernelAS(prms)
	kAS.initialize(X)
	
	kAS2 = ASI.kernelAS(prms)
	sAS = ASI.shariAS(prms)
	sAS2 = ASI.naiveShariAS(prms)

	# import IPython
	# IPython.embed()

	init_lbls = {init_pt:1}

	kAS.firstMessage(init_pt)
	fs2 = [kAS.f]

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		init_lbls[idx1] = Y[idx1]
		# sAS.setLabelCurrent(Y[idx2])
		# fs2.append(kAS.f)
		# fs3.append(sAS.f)

	print("Batch initializing:")
	print("Kernel AS:")
	kAS2.initialize(X, init_lbls)
	print("Shari AS:")
	sAS.initialize(A, init_lbls)
	print("Naive Shari AS:")
	sAS2.initialize(A, init_lbls)


	import IPython
	IPython.embed()



def RBFkernel (x,y,bw):
	x = np.array(x)
	y = np.array(y)

def test_grf ():
	bw = 10
	grf1 = 1


def test_cinv ():
	import cPickle as pickle
	import os, os.path as osp

	with open(osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/ben/Xf.pkl'),'r') as fl:
		X = pickle.load(fl)

	# X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	# X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[0]))]

	# X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	# X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[0]))]
	print X.shape
	r,n = X.shape

	nt = int(0.05*n)
	num_eval = 50
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)
	nr.shuffle(Y)

	pi = sum(Y)/len(Y)
	init_pt = 5

	# import IPython 
	# IPython.embed()

	# A = np.array((X.T.dot(X)).todense())
	t1 = time.time()

	verbose = True
	prms = ASI.Parameters(pi=pi,sparse=False, verbose=verbose)	
	kAS = ASI.kernelAS(prms)
	kAS.initialize(X)


	init_lbls = {init_pt:1}

	kAS.firstMessage(init_pt)
	# fs2 = [kAS.f]

	import IPython
	IPython.embed()

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		init_lbls[idx1] = Y[idx1]
		import IPython
		IPython.embed()


def test_CC ():

	nac = np.allclose

	n = 1000
	r = 100
	nt = 200
	rcross = 0
	X,Y = createFakeData3(n, r, nt, rcross)

	num_eval = 50
	pi = sum(Y)/len(Y)
	init_pt = 5

	# import IPython 
	# IPython.embed()

	A = X.T.dot(X)
	t1 = time.time()

	verbose = True
	prms = ASI.Parameters(pi=pi,sparse=False, verbose=verbose)	
	kAS = ASI.kernelAS(prms)
	kAS.initialize(X)
	
	sAS = ASI.shariAS(prms)
	sAS.initialize(A)
	#sAS2 = ASI.naiveShariAS(prms)

	kAS.firstMessage(init_pt)
	sAS.firstMessage(init_pt)
	# fs2 = [kAS.f]

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		# init_lbls[idx1] = Y[idx1]
		idx2 = sAS.getNextMessage()
		sAS.setLabelCurrent(Y[idx2])

		print('NEXT')
		print idx1==idx2
		print nac(kAS.f, sAS.f)
		# fs2.append(kAS.f)
		# fs3.append(sAS.f)

	import IPython 
	IPython.embed()


def test_nan ():
	import cPickle as pickle
	import os, os.path as osp

	with open(osp.join(os.getenv('HOME'), 'Research/Data/ActiveSearch/ben/forumthreadsSparseMatrix.pkl'),'r') as fl:
		X = pickle.load(fl)
		X = X.T

	# import IPython 
	# IPython.embed()
	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]

	X = X[np.squeeze(np.array(np.nonzero(X.sum(axis=1))[0])),:]
	X = X[:,np.squeeze(np.array(np.nonzero(X.sum(axis=0))[1]))]
	print X.shape
	r,n = X.shape

	nt = int(0.05*n)
	num_eval = 50
	Y = np.array([1]*nt + [0]*(n-nt), dtype=int)
	nr.shuffle(Y)

	pi = sum(Y)/len(Y)
	init_pt = 537

	# import IPython 
	# IPython.embed()

	# A = np.array((X.T.dot(X)).todense())
	t1 = time.time()

	verbose = True
	prms = ASI.Parameters(pi=pi,sparse=True, verbose=verbose)	
	kAS = ASI.kernelAS(prms)
	kAS.initialize(X)


	init_lbls = {init_pt:1}

	kAS.firstMessage(init_pt)
	# fs2 = [kAS.f]

	import IPython
	IPython.embed()

	for i in range(num_eval):
		idx1 = kAS.getNextMessage()
		kAS.setLabelCurrent(Y[idx1])
		init_lbls[idx1] = Y[idx1]
		import IPython
		IPython.embed()


if __name__ == '__main__':
	#test1(n=10, cc=2, nt=1, d=4)
	#test3()
	#testfake()
	#test4()
	#test5()
	#test6()
	# X, Y = createFakeData2(1000, 100, 300, 1)
	# Yb = Y.astype('bool')
	# Xt = X[:,Yb]
	# Xn = X[:,-Yb]
	# vis.visualize2d(Xt.T, Xn.T)
	#test7()
	#test8()
	#test9()
	# dr,h1,t1 = test10()
	# rr,h2,t2 = test11()
	#test12()
	#test13()
	# test_interface()
	# test_interface2()
	# test_interface3()
	# test_warm_start()
	# test_cinv()
	# test_CC()
	test_nan()
	# import IPython
	# IPython.embed()
	# plt.figure()
	# plt.plot(dr, h1,c='b',label='TK')
	# plt.plot(rr, h2,c='r',label='Kernel')
	# plt.xlabel('d/r')
	# plt.ylabel('hits')
	# plt.legend()
	# plt.figure()
	# plt.plot(dr, t1,c='b',label='TK')
	# plt.plot(rr, t2,c='r',label='Kernel')
	# plt.xlabel('d/r')
	# plt.ylabel('time in s')
	# plt.legend()
	# plt.show()