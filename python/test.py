from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.linalg as slg
import scipy.io as sio
import matplotlib.pyplot as plt
import time

import activeSearch as AS
from eigenmap import eigenmap
import visualize as vis

np.set_printoptions(suppress=True, precision=5, linewidth=100)

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
	# f1,_,_ = AS.kernel_AS (Xfull, Yfull, pi=pi, num_eval=num_eval, init_pt=init_pt)
	# t2 = time.time()
	Xe, b, w, deg = eigenmap(Xfull.T.dot(Xfull), d)
	# f2,_,_ = AS.lreg_AS (Xe, deg, d, alpha=0.0, labels=Yfull, options={'num_eval':num_eval,'pi':pi,'init_pt':init_pt,'n_conncomp':b})
	# t3 = time.time()

	# print "Time 1:", (t2-t1)
	# print "Time 2:", (t3-t2)


	# print f1
	# print f2

	# plt.plot(f1, color='r', label='kernel')
	# plt.plot(f2, color='b', label='lreg')
	# plt.plot(Yfull, color='g', label='true')
	# plt.legend()
	# plt.show()

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
	nt = 400
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


if __name__ == '__main__':

	#test1(n=10, cc=2, nt=1, d=4)
	#test3()
	#testfake()
	test4()
	#test5()
	#test6()
	# X, Y = createFakeData2(1000, 100, 300, 1)
	# Yb = Y.astype('bool')
	# Xt = X[:,Yb]
	# Xn = X[:,-Yb]
	# vis.visualize2d(Xt.T, Xn.T)