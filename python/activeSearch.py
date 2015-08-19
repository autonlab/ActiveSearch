from __future__ import division
import time
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

import sklearn.linear_model as slm

np.set_printoptions(suppress=True, precision=5, linewidth=100)

"""
This is an alternate implementation of active search on graphs
using a linear kernel to approximate the affinity matrix.

Choose index to reveal.
Update inverse operation.
Update f after this inverse is computed.

"""
def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))


def kernel_AS (X, labels, num_initial=1, num_eval=1, pi=0.05, eta=0.5, w0=None, alpha=0.2, init_pt=None, verbose=True, all_fs=False, sparse=False, tinv=False):
	"""
	X 			--> r x n matrix of feature values for each point.
	labels 		--> true labels for each point.
	pi 			--> prior target probability
	eta 		--> jump probability
	num_initial --> number of initial points to start from.
	num_eval 	--> number of points to be investigated
	"""
	#X = np.array(X)
	r,n = X.shape
	labels = np.array(labels)

	if init_pt is not None:
		if not isinstance(init_pt, list):
			init_pt = [init_pt]

		num_initial = len(init_pt)
		idxs = init_pt
		if not (labels[idxs]).all():
			if verbose:
				print "Warning: start points provided not targets. Converting to targets."
			labels[idxs] = 1
			true_targets = (np.array(labels)==1).nonzero()[0]
			
		unlabeled_idxs = [i for i in range(n) if i not in idxs]
	else:
		# Random start node
		true_targets = (np.array(labels)==1).nonzero()[0]
		# %%% Randomly pick 1 target point as the first point
		idxs = [true_targets[i] for i in nr.permutation(range(len(true_targets)))[:num_initial]]
		unlabeled_idxs = [i for i in range(n) if i not in idxs]
		# %%% Randomly pick 1 target point as the first point

	num_initial = min(n-1,num_initial)
	num_eval = min(num_eval, n-num_initial)

	if sparse:
		Ir = ss.eye(r)
	else:
		Ir = np.eye(r)

	# Lambda from TK's notes can be written using eta as follows
	l = (1-eta)/eta
	# omega0 as in TK's code
	if w0 is None: w0 = 1/n

	# Set up initial BD and C
	B = 1/(1+w0)*np.ones(n) # Need to update B every iteration
	B[idxs] = l/(1+l)
	D = np.squeeze(X.T.dot(X.dot(np.ones((n,1))))) #TODO: if we don't need to keep this, we can remove it.
	Dinv = 1./D
	BDinv = np.squeeze(B*Dinv)

	if sparse:
		BDinv_ss = ss.diags([BDinv],[0]).tocsr()

	if verbose:
		print 'Start point: \n', idxs

	y = pi*np.ones(n)
	y[idxs] = 1
	I_B = 1-B
	q = I_B*y # Need to update q every iteration

	# the atleast 2d transpose is there to make it be the same as diag(__)

	# import IPython
	# IPython.embed()
	if verbose: 
		print "Constructing C"
		t1 = time.time()

	if sparse:
		C = (Ir - X.dot(BDinv_ss.dot(X.T)))	
	else:
		C = (Ir - X.dot(BDinv[:,None]*X.T))

	if verbose:
		print time.time() - t1

	# import IPython
	# IPython.embed()

	if verbose:
		print "Inverting"
	t1 = time.time()
	if sparse:
		Cinv = ssl.inv(C.tocsc()) # Need to update Cinv every iteration
	else:
		Cinv = nlg.inv(C)

	dtinv = time.time() - t1
	if verbose:
		print "Time for inverse:", dtinv


	hits = np.zeros((num_eval+num_initial,1))
	hits[0] = num_initial
	selected = [ix for ix in idxs]

	if sparse:
		f = q + BDinv_ss.dot(((X.T.dot(Cinv.dot(X.dot(q))))))
	else:
		f = q + BDinv*((X.T.dot(Cinv.dot(X.dot(q)))))

	# Computations for the impact factor:
	
	# 0. Some useful variables 
	S = w0*np.ones(n)
	S[idxs] = 1/(1+l)
	I_BPinv = np.squeeze(I_B*(1/(S*D))) #(I-B)*Pinv
	dP = (1./l-w0)*D # L - U
	dPpi = (1./l-pi*w0)*D # L - pi*U
	
	# 1. Df_tilde
	# First, we need J = diag (X^T * Cinv * X): each element of J is x_i^T*Cinv*x_i
	J = np.squeeze(((Cinv.dot(X))*X).sum(0))
	# Now we compute the entire diag
	diagM = (1+BDinv*J)*I_BPinv
	# Finally, Df_tilde
	dpf = (dPpi - dP*f)
	Df_tilde = dpf*diagM/(1 + dP*diagM)

	# 2. DF
	# z = (I-B)Pinv*u (these are defined in Kernel AS notes)
	z = I_BPinv.copy()
	z[idxs] = 0
	Minv_u = z + BDinv*((X.T.dot(Cinv.dot(X.dot(z)))))
	DF = (dpf - dP*Df_tilde)*Minv_u

	# 3. IM
	IM = f*(DF-Df_tilde)

	# Number of true targets
	true_n = sum(labels==1)
	found_n = num_initial

	if all_fs:
		fs = [f]

	# Modifying the element 
	for i in range(num_eval):

		# import IPython
		# IPython.embed()
		t1 = time.time()

		# assert len(unlabeled_idxs) == n - num_initial - i
		# if len(unlabeled_idxs) != len(np.unique(unlabeled_idxs)):
		# 	print "ERROR: NOT ALL UNLABELED IDXS ARE UNIQUE"

		# Find next index to investigate
		uidx = np.argmax((f+alpha*IM)[unlabeled_idxs])
		idx = unlabeled_idxs[uidx]
		# if idx == n:
		# 	print "ERROR: SELECTING SELECTED PT", idx
		# 	import IPython
		# 	IPython.embed()

		del unlabeled_idxs[uidx]

		# assert idx not in unlabeled_idxs

		found_n += labels[idx]
		if found_n==true_n:
			if verbose:
				print "Found all", found_n, "targets. Breaking out."
			break

		# Update relevant matrices

		if sparse:
			BDinv_ss[idx,idx] *= (1+w0)*l/(1+l) 
		else:
			BDinv[idx] *= (1+w0)*l/(1+l) 

		q[idx] = labels[idx]*1/(1+l)
		gamma = -(l/(1+l)-1/(1+w0))*Dinv[idx]

		Xi = X[:,[idx]] # ith feature vector

		# t7 = time.time()
		Cif = Cinv.dot(Xi)
		# t8 = time.time()
		# d7 = t8 - t7
		# import IPython
		# IPython.embed()
		if sparse:
			c =  gamma/(1 + (gamma*Xi.T.dot(Cif))[0,0])
		else:
			c =  gamma/(1 + gamma*Xi.T.dot(Cif))
		Cinv = Cinv - c*(Cif.dot(Cif.T))
		
		# t9 = time.time()
		# d8 = t9 - t8

		if sparse:
			f = q + BDinv_ss.dot(((X.T.dot(Cinv.dot(X.dot(q))))))
		else:
			f = q + BDinv*((X.T.dot(Cinv.dot(X.dot(q)))))

		# Updating IM
		z[idx] = 0
		Minv_u = z + BDinv*((X.T.dot(Cinv.dot(X.dot(z)))))
		dpf = (dPpi - dP*f)*Minv_u
		# Updating Df_tilde
		J = J - c*[(X.T*Cif)**2]
		diagM = (1+BDinv*J)*I_BPinv
		Df_tilde = dpf*diagM/(1 + dP*diagM)
		# Updating DF
		DF = (dpf - dP*Df_tilde)*Minv_u
		# Computing IM
		IM = f*(DF-Df_tilde)

		if all_fs:
			fs.append(f)

		# import IPython
		# IPython.embed()

		elapsed = time.time() - t1
		selected.append(idx)
		hits[i+1] = found_n

		## temp ##
		# B2[idx] = l/(l+1)
		# yp[idx] = float(labels[idx])
		# Ap = np.diag(B2).dot(dinvA)
		# q2 = (np.eye(n) - np.diag(B2)).dot(yp)
		# f2 = nlg.inv(np.eye(n) - Ap).dot(q2)
		# print nlg.norm(f-f2)
		## temp ##

		if verbose:
			if (i%1)==0 or i==1:
				print 'Iter: %i, Selected: %i, Best f: %f, Hits: %i/%i, Time: %f'%(i,selected[i+num_initial], f[idx], hits[i+1], (i+num_initial+1), elapsed)
			print '%d %d %f %d\n'%(i, hits[i+1]/true_n, elapsed, selected[i+num_initial])


	# Ap = np.diag(B2).dot(dinvA)
	# q2 = (np.eye(n) - np.diag(B2)).dot(yp)
	# f2 = nlg.inv(np.eye(n) - Ap).dot(q2)
	if all_fs:
		if tinv: return f, hits, selected, fs, dtinv
		return f, hits, selected, fs
	if tinv: return f, hits, selected, dtinv
	return f, hits, selected

def lreg_AS (X, deg, dim, alpha, labels, options={}, verbose=True):
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
	print X.shape
	n,d = X.shape
	labels = np.array(labels)
	true_targets = (np.array(labels)==1).nonzero()[0]

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
	if 'init_pt' in options:
		init_pt = options['init_pt']
		if not isinstance(init_pt, list):
			init_pt = [init_pt]

		num_initial = len(init_pt)
		start_point = init_pt
		if not (labels[start_point]).all():
			if verbose:
				print "Warning: start points provided not targets. Converting to targets."
			labels[start_point] = 1
			true_targets = (np.array(labels)==1).nonzero()[0]
			yp = labels[:,None] * sqd
	else:
		num_initial = 1 # For now we always initialize with 1 target point.
		# %%% Randomly pick 1 target point as the first point
		start_point = [true_targets[i] for i in nr.permutation(range(len(true_targets)))[:num_initial]]

	in_train = np.zeros((n,1)).astype('bool')
	in_train[start_point] = True
	best_ind = start_point

	l = (1-eta)/eta
	r = l*w0
	c = 1/(1-r)
	Xp = X*sqd

	if verbose:
		print 'Start point: \n', best_ind
	hits = np.zeros((num_eval+1,1))
	selected = [ix for ix in best_ind]
	hits[0] = num_initial

	C = r*(Xp.T.dot(Xp)) + (1-r)*Xp[best_ind,:].T.dot(Xp[best_ind,:]) + l*np.diag([0]*b+[1]*(d-b))
	Cinv = nlg.inv(C)

	h = (Xp.dot(Cinv)*Xp).sum(axis=1)[:,None]
	f = X.dot(Cinv.dot(r*Xp.T.dot(sqd)*pi + Xp[best_ind,:].T.dot(yp[best_ind]-r*sqd[best_ind]*pi)))

	# Number of true targets
	true_n = sum(labels==1)
	found_n = num_initial

	# %%% Main loop
	if verbose:
		print "Entering main loop"
	for i in range(num_eval):

		t1 = time.time()
		# %%% Calculating change
		in_test = (-in_train)
		test_ind = in_test.astype('int')
		change = ((test_ind.T.dot(X).dot(Cinv).dot(Xp.T).T - (h/sqd))* sqd *((1-r*pi)*c-f))/ (c+h)

		f_bnd = np.squeeze(np.minimum(np.maximum(f[in_test],0),1))[:,None]
		 
		# %%% Calculating selection criteria
		score = f_bnd + alpha*f_bnd*np.squeeze(np.maximum(change[in_test],0))[:,None]

		# %%% select best index
		best_ind = np.argmax(score)
		best_score = score[best_ind]
		best_f = f_bnd[best_ind]
		test_ind = in_test.nonzero()[0]
		best_ind = test_ind[best_ind]
		best_change = max(np.max(change[best_ind]),0)
		in_train[best_ind] = True
		
		found_n += labels[best_ind]
		if found_n==true_n:
			if verbose:
				print "Found all", found_n, "targets. Breaking out."
			break
		# %%% Updating parameters
		# %keyboard;
		# %yp(best_ind) = sqd(best_ind) * input(['Requesting label for e-mail '  num2str(best_ind)  ':']);
		CXp = Cinv.dot(Xp[[best_ind],:].T)
		f2 = f
		f = f + X.dot(  CXp*((yp[best_ind]-r*sqd[best_ind]*pi)*c - sqd[best_ind]*f[best_ind]) / (c+h[best_ind])  )

		# %f = f + X * CXp * (yp_new(i) - yp(i));
		Cinv = Cinv - (CXp.dot(CXp.T))/(c+h[best_ind])
		h = h - (Xp.dot(CXp)**2)/(c+h[best_ind])

		elapsed = time.time() - t1

		selected.append(best_ind)
		hits[i+1] = found_n
		if verbose:
			if (i%1)==0 or i==1:
				print 'Iter: %i, Selected: %i, E[u]: %f, Best f: %f, Best change: %f, Hits: %i/%i, Time: %f'%(i+1,selected[i+num_initial], best_score, best_f, best_change, hits[i+1], (i+num_initial+1), elapsed)
			print '%d %d %f %d\n'%(i+1, hits[i+1], elapsed, selected[i+num_initial])

	return np.squeeze(f), hits, selected


def shari_activesearch_probs_naive(A, labels, pi, num_eval, w0=None, eta=None, init_pt=None, verbose=False, all_fs=False):
	'''
	Gets the vector of label probabilities for active search.
	
	Parameters
	----------
	
	A: array, shape [n, n]
		The $A$ matrix of similarity scores.
	
	labels: array, shape n
		The labels for each node. 0 means negative, 1 means positive,
		negative value means unobserved.
		
	pi: scalar in [0, 1]
		Prior probability of being positive.
		
	w0: nonnegative scalar
		Strength of that prior.

	other variables are simple
	'''
	D = A.sum(axis=1)
	n, = D.shape
	labels = np.array(labels)
	true_targets = (np.array(labels)==1).nonzero()[0]

	if eta is None:
		eta = 0.5
	if w0 is None:
		w0 = 1/n
	if init_pt is not None:
		if not isinstance(init_pt, list):
			init_pt = [init_pt]
		num_initial = len(init_pt)
		start_point = init_pt
		if not (labels[start_point]).all():
			if verbose:
				print "Warning: start points provided not targets. Converting to targets."
			labels[start_point] = 1
			true_targets = (np.array(labels)==1).nonzero()[0]
	else:
		num_initial = 1 # For now we always initialize with 1 target point.
		# %%% Randomly pick 1 target point as the first point
		start_point = [true_targets[i] for i in nr.permutation(range(len(true_targets)))[:num_initial]]
	unlabeled_idxs = [i for i in range(n) if i not in start_point]


	lam = (1-eta)/eta

	hits = np.zeros((num_eval+num_initial,1))
	hits[0] = num_initial
	selected = [ix for ix in start_point]

	#wts = np.where(labeled, lam / (1 + lam), 1 / (1 + w0))
	wts = 1/(1+w0)*np.ones(n)
	wts[start_point] = lam / (1 + lam)

	I_minus_Ap = (-wts / D)[:, None] * A
	I_minus_Ap[xrange(n), xrange(n)] += 1
	yp = np.ones(n)*pi
	yp[start_point] = float(labels[start_point])
	Dp_yp = (1 - wts) * yp
	f =  np.linalg.solve(I_minus_Ap, Dp_yp)

	# Number of true targets
	true_n = sum(labels==1)
	found_n = num_initial

	if all_fs:
		fs = [f]
	
	for i in range(num_eval):
		
		# import IPython
		# IPython.embed()

		t1 = time.time()

		uidx = np.argmax(f[unlabeled_idxs])
		idx = unlabeled_idxs[uidx]

		del unlabeled_idxs[uidx]

		wts[idx] = lam/(1+lam)
		I_minus_Ap = (-wts / D)[:, None] * A
		I_minus_Ap[xrange(n), xrange(n)] += 1
		yp[idx] = labels[idx]
		Dp_yp = (1 - wts) * yp
		f =  np.linalg.solve(I_minus_Ap, Dp_yp)

		if all_fs:
			fs.append(f)

		elapsed = time.time() - t1
		selected.append(idx)
		hits[i+1] = found_n

		if verbose:
			if (i%1)==0 or i==1:
				print 'Iter: %i, Selected: %i, Best f: %f, Hits: %i/%i, Time: %f'%(i,selected[i+num_initial], f[idx], hits[i+1],(i+num_initial+1), elapsed)
			print '%d %d %f %d\n'%(i, hits[i+1], elapsed, selected[i+num_initial])

	if all_fs:
		return f, hits, selected, fs
	return f, hits, selected


def LinearKernel (x,y):
	return np.squeeze(np.array(x)).dot(np.squeeze(np.array(y)))

def AbsDifference (x,y):
	return np.squeeze(np.abs(np.array(x)-np.array(y)))

def PointWiseMultiply (x,y):
	return np.squeeze(matrix_squeeze(x)*matrix_squeeze(y))


def fit_regressor (Xf, labels, logit=None, C=1, reg='l2', sfunc='LinearKernel'):
	"""
	Xf: 		feature vectors (r x n) where r is the number of features and n is the number of data points
	labels:		known labels -- doesn't need to be all of the points
	C:			inverse of regularization parameter for l1/l2 regularizer
	sfunc:		similarity function between points

	Fits a logistic regressor such that g(sfunc(xi.T, xj)) > 0 if i,j is in same class and < 0 otherwise.
	"""
	# if sfunc != LinearKernel:
	# 	raise NotImplementedError('Only linear kernel works for now')

	Xf = Xf
	r,n = Xf.shape
	nl = len(labels)
	keys = labels.keys()
	
	Xf_labeled = Xf[:,keys]
	Xf_unlabeled = Xf[:,list(set(xrange(n)) - set(keys))]

	lvals = np.atleast_2d(labels.values())

	inputX = []
	inputY = []

	if sfunc == 'LinearKernel':
		Af_labeled = Xf_labeled.T.dot(Xf_labeled)[np.triu_indices(nl)].tolist()
		Af_unlabeled = (Xf_unlabeled**2).sum(0).tolist()
		Yf_labeled = lvals.T.dot(lvals)[np.triu_indices(nl)].tolist()
		Yf_unlabeled = [1]*(n-nl)

		inputX = Af_labeled + Af_unlabeled
		inputY = Yf_labeled + Yf_unlabeled
		inputX = np.atleast_2d(inputX).T

	elif sfunc == 'AbsDifference':
		Af_labeled = []
		Yf_labeled = lvals.T.dot(lvals)[np.triu_indices(nl,1)].tolist()
		for i,j in zip(*np.triu_indices(nl,1)):
			Af_labeled.append(AbsDifference(Xf_labeled[:,i],Xf_labeled[:,j]))
		Af_unlabeled = [np.zeros(r)]
		Yf_unlabeled = [1]

		inputX = Af_labeled + Af_unlabeled
		inputY = Yf_labeled + Yf_unlabeled

	elif sfunc == 'PointWiseMultiply':
		Af_labeled = []
		Yf_labeled = lvals.T.dot(lvals)[np.triu_indices(nl)].tolist()
		for i,j in zip(*np.triu_indices(nl)):
			Af_labeled.append(PointWiseMultiply(Xf_labeled[:,i],Xf_labeled[:,j]))
		Af_unlabeled = (Xf_unlabeled.T**2).tolist()
		Yf_unlabeled = [1]*(n-nl)

		inputX = Af_labeled + Af_unlabeled
		inputY = Yf_labeled + Yf_unlabeled

	# Train the regressor
	if logit is None:
		logit = slm.LogisticRegression (penalty=reg, C=C)

	inputX = np.asarray(inputX)
	logit.fit(inputX, inputY)

	return logit

use_prob = True

def regression_active_search (X, labels, num_eval=100, w0=None, pi=0.05, eta=0.5, alpha=1.0, sfunc='LinearKernel', C=1.0, init_pt=None, verbose=True, all_fs=False, sparse=False):
	"""
	X 			--> r x n matrix of feature values for each point.
	labels 		--> true labels for each point.
	pi 			--> prior target probability
	num_eval 	--> number of points to be investigated
	"""
	#X = np.array(X)
	r,n = X.shape
	labels = np.array(labels)

	# Random start node
	true_positives = (np.array(labels)==1).nonzero()[0]
	true_negatives = (np.array(labels)==0).nonzero()[0]
	# %%% Randomly pick 1 target point as the first point
	idxs = [true_positives[nr.randint(len(true_positives))], true_negatives[nr.randint(len(true_negatives))]]
	unlabeled_idxs = [i for i in range(n) if i not in idxs]

	num_initial = 2
	
	if w0 is None: w0 = 1/n
	C = 1.0

	f = np.zeros(n)

	known_labels = {idx:{0:-1,1:1}[labels[idx]] for idx in idxs}

	bval = eta/(1-eta)
	beta = 1 - w0/bval

	logit = None

	hits = np.zeros((num_eval+num_initial,1))
	hits[0] = 1
	hits[1] = 1
	selected = [ix for ix in idxs]
	# Number of true targets
	true_n = sum(labels==1)
	found_n = 1

	if all_fs:
		fs = [f]


	# Modifying the element 
	for i in range(num_eval):

		# import IPython
		# IPython.embed()
		t1 = time.time()

		# assert len(unlabeled_idxs) == n - num_initial - i
		# if len(unlabeled_idxs) != len(np.unique(unlabeled_idxs)):
		# 	print "ERROR: NOT ALL UNLABELED IDXS ARE UNIQUE"

		# Find next index to investigate
		uidx = np.argmax(f[unlabeled_idxs])
		idx = unlabeled_idxs[uidx]

		known_labels[idx] = {0:-1,1:1}[labels[idx]]

		del unlabeled_idxs[uidx]

		# assert idx not in unlabeled_idxs

		found_n += labels[idx]
		if found_n==true_n:
			if verbose:
				print "Found all", found_n, "targets. Breaking out."
			break

		logit = fit_regressor(X, known_labels, logit=logit, C=C, reg='l2', sfunc=sfunc)
		# Update relevant matrices
		if sfunc =='LinearKernel':
			x_i = np.atleast_2d(X.T.dot(X[:,idx])).T
		elif sfunc == 'AbsDifference':
			x_i = np.abs(X.T-X[:,idx])
		elif sfunc == 'PointWiseMultiply':
			x_i = X.T*	X[:,idx]

		if use_prob:
			delta_f = logit.predict_proba(x_i).dot([1,-1])
		else:
			delta_f = logit.predict(x_i)


		c  = ((labels[idx] - pi) + beta*(pi - f[idx]))*bval*sum(x_i)[0]

		f = f + {0:-1,1:1}[labels[idx]]*delta_f*alpha

		# t0 = time.time()
		# d9 = t0 - t9

		if all_fs:
			fs.append(f)

		# import IPython
		# IPython.embed()

		elapsed = time.time() - t1
		selected.append(idx)
		hits[i+1] = found_n

		## temp ##
		# B2[idx] = l/(l+1)
		# yp[idx] = float(labels[idx])
		# Ap = np.diag(B2).dot(dinvA)
		# q2 = (np.eye(n) - np.diag(B2)).dot(yp)
		# f2 = nlg.inv(np.eye(n) - Ap).dot(q2)
		# print nlg.norm(f-f2)
		## temp ##

		if verbose:
			if (i%1)==0 or i==1:
				print 'Iter: %i, Selected: %i, Best f: %f, Hits: %i/%i, Time: %f'%(i,selected[i+num_initial], f[idx], hits[i+1], (i+num_initial+1), elapsed)
			print '%d %d %f %d\n'%(i, hits[i+1]/true_n, elapsed, selected[i+num_initial])


	# Ap = np.diag(B2).dot(dinvA)
	# q2 = (np.eye(n) - np.diag(B2)).dot(yp)
	# f2 = nlg.inv(np.eye(n) - Ap).dot(q2)
	if all_fs:
		return f, hits, selected, fs, logit
	return f, hits, selected, logit