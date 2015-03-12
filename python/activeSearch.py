from __future__ import division
import time
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg

from eigenmap import eigenmap

np.set_printoptions(suppress=True, precision=3, linewidth=100)

"""
This is an alternate implementation of active search on graphs
using a linear kernel to approximate the affinity matrix.

Choose index to reveal.
Update inverse operation.
Update f after this inverse is computed.

"""


def kernel_AS (X, labels, num_initial=1, num_eval=1, pi=0.05, eta=0.5, w0=None):
	"""
	X 			--> r x n matrix of feature values for each point.
	labels 		--> true labels for each point.
	pi 			--> prior target probability
	eta 		--> jump probability
	num_initial --> number of initial points to start from.
	num_eval 	--> number of points to be investigated
	"""
	X = np.array(X)
	r,n = X.shape
	labels = np.array(labels)

	num_initial = min(n-1,num_initial)
	num_eval = min(num_eval, n-num_initial)

	Ir = np.eye(r)

	# Lambda from TK's notes can be written using eta as follows
	l = (1-eta)/eta
	# omega0 as in TK's code
	if w0 is None: w0 = 1/n

	# Random start node
	true_targets = (np.array(labels)==1).nonzero()[0]
	idxs = [true_targets[i] for i in nr.permutation(range(len(true_targets)))[:num_initial]]
	unlabeled_idxs = [i for i in range(n) if i not in idxs]
	# Set up initial BD and C
	B = 1/(1+w0)*np.ones(n) # Need to update B every iteration
	B[idxs] *= (1+w0)*l/(1+l)
	D = np.squeeze(X.T.dot(X.dot(np.ones((n,1))))) #TODO: if we don't need to keep this, we can remove it.
	Dinv = 1./D
	BDinv = np.squeeze(B*Dinv)
	#BDinv = np.atleast_2d(B*Dinv).T
	#BDinvMat = ss.diags([BDinv], [0]) 

	y = pi*np.ones(n)
	y[idxs] = 1
	I_B = 1-B
	q = I_B*y # Need to update q every iteration

	# the atleast 2d transpose is there to make it be the same as diag(__)
	C = (Ir - X.dot(BDinv[:,None]*X.T)) 
	Cinv = nlg.inv(C) # Need to update Cinv every iteration

	hits = np.zeros((num_eval+num_initial,1))
	selected = np.zeros((num_eval+num_initial,1))
	hits[0] = 1
	selected[:num_initial] = idxs

	# Number of true targets
	true_n = sum(labels==1)
	found_n = num_initial

	# Modifying the element 
	for i in range(num_eval):

		t1 = time.time()

		f = q + BDinv*((X.T.dot(Cinv.dot(X.dot(q)))))

		# Find next index to investigate
		uidx = np.argmax(f[unlabeled_idxs])
		idx = unlabeled_idxs[uidx]
		del unlabeled_idxs[uidx]

		found_n += labels[idx]
		if found_n==true_n:
			print "Found all", found_n, "targets. Breaking out."
			break

		# Update relevant matrices
		BDinv[idx] *= (1+w0)*l/(1+l) 
		q[idx] = labels[idx]*l/(1+l)
		gamma = -(l/(1+l)-1/(1+w0))*Dinv[idx]

		Xi = X[:,[idx]] # ith feature vector
		Cif = Cinv.dot(Xi)
		Cinv = Cinv - gamma*(Cif.dot(Cif.T))/(1 + gamma*Xi.T.dot(Cif))

		elapsed = time.time() - t1
		selected[i+num_initial] = idx
		hits[i+1] = found_n

		if (i%1)==0 or i==1:
			print 'Iter: %i, Selected: %i, Best f: %f, Hits: %f, Time: %f'%(i,selected[i+num_initial], f[idx], hits[i+1]/(i+num_initial+1), elapsed)
		print '%d %d %f %d\n'%(i, hits[i+1], elapsed, selected[i+num_initial])

	return f, hits, selected

def lreg_AS (X, deg, dim, alpha, labels, options={}):
	# %%% [hits,selected] = lreg_AS_main(X,deg,dim,alpha,labels,options) 
	# %%% Input: 
	# %%% X: n-by-d matrix. Each row is the feature vector of a node computed by Eigenmap.
	# %%% deg: n-by-1 vector. Each entry is the sum of pairwise similarity values between a node and all other nodes.
	# %%% dim: a positive integer indicating the number of leading dimensions in X to use
	# %%% alpha: a positive real number used in the calculation of the selection score
	# %%% labels: n-by-1 vector. True labels of data points. 1: target, 0: non-target.
	# %%% options: a structure specifying values for the following algorithmic options:
	# %%% 	num_evaluations: number of points we want to investigate (default: 5000)
	# %%% 	randomSeed: seed of random number generator (default: the current time)
	# %%% 	log_prefix: prefix of log file name (default: current time string)
	# %%% 	n_conncomp: number of connected components in the similarity matrix (default 1)
	# %%% 	omega0: weight assigned to regularization on unlabeled points (default: 1/(#data points))
	# %%% 	prior_prob: prior target probability (default: 0.05)
	# %%% 	eta: jump probability (default: 0.5)
	# %%%
	# %%% Output:
	# %%% hits: a vector of cumulative counts of discovered target points
	# %%% selected: a vector of indices of points selected by the algorithm 

	X = X[:,:dim]
	n,d = X.shape
	labels = np.array(labels)

	sqd = np.sqrt(deg)[:,None]
	yp = labels[:,None] * sqd

	# %%% Default values for options
	# num_evaluations - number of points we want to investigate
	# b - number of connected components in the similarity matrix
	# w0 - weight assigned to regularization on unlabeled points
	# pi - prior target probability
	# eta - jump probability
	if 'num_eval' in options:
		num_eval = options['num_eval']
	else: num_eval = 5
	if 'n_conncomp' in options:
		b = options['n_conncomp']
	else: b = 1
	if 'w0' in options:
		w0 = options['w0']
	else: w0 = 1/n
	if 'pi' in options:
		pi = options['pi']
	else: pi = 0.05
	if 'eta' in options:
		eta = options['eta']
	else: eta = 0.5

	num_initial = 1 # For now we always initialize with 1 target point.
	l = (1-eta)/eta
	r = l*w0
	c = 1/(1-r)

	# %%% Randomly pick 1 target point as the first point
	Xp = X*sqd
	true_targets = (np.array(labels)==1).nonzero()[0]
	start_point = [true_targets[i] for i in nr.permutation(range(len(true_targets)))[:num_initial]]
	in_train = np.zeros((n,1)).astype('bool')
	in_train[start_point] = True
	best_ind = start_point

	print 'Start point: \n', best_ind
	hits = np.zeros((num_eval+1,1))
	selected = np.zeros((num_eval+num_initial,1))
	hits[0] = num_initial
	selected[:num_initial] = best_ind


	C = r*(Xp.T.dot(Xp)) + (1-r)*(Xp[best_ind,:].T.dot(Xp[best_ind,:]) + l*np.diag([0]*b+[1]*(d-b)))
	Cinv = nlg.inv(C)

	h = (Xp.dot(Cinv)*Xp).sum(axis=1)[:,None]
	f = X.dot(Cinv.dot(r*Xp.T.dot(sqd)*pi + Xp[best_ind,:].T.dot(yp[best_ind]-r*sqd[best_ind]*pi)))

	# Number of true targets
	true_n = sum(labels==1)
	found_n = num_initial

	# %%% Main loop
	for i in range(num_eval):

		t1 = time.time()
		# %%% Calculating change
		test_ind = (-in_train).astype('int') 
		change = ((test_ind.T.dot(X).dot(Cinv).dot(Xp.T).T - (h/sqd))* sqd *((1-r*pi)*c-f))/ (c+h)

		# import IPython
		# IPython.embed()

		f_bnd = np.squeeze(np.minimum(np.maximum(f[test_ind],0),1))[:,None]
		 
		# %%% Calculating selection criteria
		score = f_bnd + alpha*f_bnd*np.squeeze(np.maximum(change[test_ind],0))[:,None]

		# import IPython
		# IPython.embed()

		# %%% select best index
		best_ind = np.argmax(score)
		best_score = score[best_ind]
		best_f = f_bnd[best_ind]
		test_ind = test_ind.nonzero()[0]
		best_ind = test_ind[best_ind]
		best_change = max(np.max(change[best_ind]),0)
		in_train[best_ind] = True
		
		found_n += labels[best_ind]
		if found_n==true_n:
			print "Found all", found_n, "targets. Breaking out."
			break

		# %%% Updating parameters
		# %keyboard;
		# %yp(best_ind) = sqd(best_ind) * input(['Requesting label for e-mail '  num2str(best_ind)  ':']);
		CXp = Cinv.dot(Xp[[best_ind],:].T)
		f = f + X.dot(  CXp*((yp[best_ind]-r*sqd[best_ind]*pi)*c - sqd[best_ind]*f[best_ind]) / (c+h[best_ind])  )
		# %f = f + X * CXp * (yp_new(i) - yp(i));
		Cinv = Cinv - (CXp.dot(CXp.T))/(c+h[best_ind])
		h = h - (Xp.dot(CXp)**2)/(c+h[best_ind])
		
		elapsed = time.time() - t1

		selected[i+num_initial] = best_ind
		hits[i+1] = found_n
		if (i%1)==0 or i==1:
			print 'Iter: %i, Selected: %i, E[u]: %f, Best f: %f, Best change: %f, Hits: %f, Time: %f'%(i,selected[i+num_initial], best_score, best_f, best_change, hits[i+1]/(i+num_initial+1), elapsed)
		print '%d %d %f %d\n'%(i, hits[i+1], elapsed, selected[i+num_initial])

	return f, hits, selected


def test1 (n, cc=2, nt=1, d=5):
	# Created banded diagonal kernel matrix
	X = np.eye(n)
	X[range(n-1), range(1,n)] = 1
	Xfull = slg.block_diag(*([X]*cc))
	Yfull = [1]*(nt*n) + [0]*((cc-nt)*n)

	Xe, b, w, deg = eigenmap(Xfull.T.dot(Xfull), d)

	f1 = kernel_AS (Xfull, Yfull, pi=(nt*1.0)/cc, num_eval=10)

	f2 = lreg_AS (Xe, deg, d, alpha=0.0, labels=Yfull, options={'num_eval':3,'pi':(nt*1.0)/cc})

	import IPython
	IPython.embed()


if __name__ == '__main__':

	test1(10, cc=2, nt=1)
