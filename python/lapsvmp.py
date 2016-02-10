# Ported from Matlab to Python by Sibi Venkatesan (sibiv@andrew.cmu.edu)
# with help from SMOP version 
# /usr/local/bin/smop lapsvmp.m

# % {lapsvmp} trains a Laplacian SVM classifier in the primal.
# %     
# %      classifier = lapsvmp(options,data)
# %
# %      options: a structure with the following fields
# %               options.gamma_A: regularization parameter (ambient norm)
# %               options.gamma_I: regularization parameter (intrinsic norm)
# %
# %               [optional fields]
# %               options.Cg: {0,1} i.e. train with Newton's method or PCG
# %                           (default=0)
# %               options.MaxIter: maximum number of iterations (default=200)
# %               options.Hinge: {0,1} i.e. train a LapSVM (1) or LapRLSC (0)
# %                              (default=1)
# %               options.UseBias: {0,1} i.e. use or not a bias (default=0)
# %               options.InitAlpha: if it's 0, the initial weights are null;
# %                                  if it's <0, they are randomly taken;
# %                                  otherwise it is the initial vector
# %                                  (default=0)
# %               options.InitBias: the initial bias (default=0)
# %               options.NewtonLineSearch: {0,1} i.e. use or not exact line
# %                                         search with Newton's method
# %                                         (default=0).
# %               options.NewtonCholesky: {0,1} i.e. use or not Cholesky
# %                                       factorization and rank 1 updates of
# %                                       the Heassian (default=0 for LapSVM.
# %                                       If set to 1, the  Hessian must be
# %                                       positive definite). 
# %               options.CgStopType: {0,1,2,3,4,5,6,7,-1}
# %                                   the data-based stopping criterion of cg
# %                                   only (default=0), where:
# %                                   0: do not stop
# %                                   1: stability stop
# %                                   2: validation stop
# %                                   3: stability & validation stop
# %                                   4: gradient norm (normalized)
# %                                   5: prec. gradient norm (normalized)
# %                                   6: mixed gradient norm (normalized)
# %                                   7: relative objective function change
# %                                  -1: debug (it saves many stats)
# %               options.CgStopIter: number of cg iters after which check
# %                                   the stopping condition.
# %               options.CgStopParam: the parameter for the selected
# %                                    CgStopType. If CgStopType is:
# %                                    0: ignored
# %                                    1: percentage [0,1] of tolerated
# %                                       different decisions between two
# %                                       consecutive checks
# %                                    2: percentage [0,1] of error rate
# %                                       decrement required  between two
# %                                       consecutive checks
# %                                    3: the two params above (i.e. it is an
# %                                       array of two elements)
# %                                    4: the minimum gradient norm
# %                                    5: the minimum prec. gradient norm 
# %                                    6: the minimum mixed gradient norm
# %                                    7: relative objective function change
# %                                       between two consecutive checks
# %                                    -1: ignored
# %                                    see the code for default values.
# %               options.Verbose: {0,1} (default=1)
# %
# %      data: a structure with the following fields
# %            data.X: a N-by-D matrix of N D-dimensional training examples
# %            data.K: a N-by-N kernel Gram matrix of N training examples
# %            data.Y: a N-by-1 label vector in {-1,0,+1}, where 0=unlabeled
# %                    (it is in {0,+1} in the case of One-Class SVM/LapSVM.
# %            data.L: a N-by-N matrix of the Laplacian
# %
# %            [other fields]
# %            data.Kv: a V-by-N kernel Gram matrix of V validation examples,
# %                     required if CgStopType is 2 or 3 (validation check)
# %            data.Yv: a V-by-1 vector of {-1,+1} labels for the validation
# %                     examples, required if CgStopType is 2 or 3
# %                     (validation check)
# %
# %      classifier: structure of the trained classifier (see the
# %                  'saveclassfier' function). 
# %
# % Author: Stefano Melacci (2012)
# %         mela@dii.unisi.it
# %         * the One-Class extension is joint work with Salvatore Frandina,
# %           salvatore.frandina@gmail.com
# %         * the original code structure was based on the primal SVM code of 
# %           Olivier Chapelle, olivier.chapelle@tuebingen.mpg.de 
from __future__ import division
import time
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

import graph_utils as gu

import IPython

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_squeeze(X):
	# converts into numpy.array and squeezes out singular dimensions
	return np.squeeze(np.asarray(X))


class Classifier:
	def __init__(self, name, svs, alpha, b, xtrain, options, sec=None, t=None, lsiters=None, stats=None):
		self.name = name
		self.svs = svs
		self.alpha = alpha
		self.b = b
		self.xtrain = xtrain
		self.options = options

		self.t = t
		self.lsiters = lsiters
		self.traintime = sec
		self.stats = stats

class LapSVMOptions (object):
	def __init__ (self):
		self.Verbose =True
		self.Sparse = False

		self.Kernel = 'rbf'
		self.KernelParam = 1
		self.NN = 6
		self.GraphDistanceFunction = 'euclidean'
		self.GraphWeights = 'heat'
		self.GraphWeightParam = 0
		self.LaplacianNormalize = True
		self.LaplacianDegree = 1
		self.gamma_A = 1e-6
		self.gamma_I = 1.0
		self.Hinge = True
		self.Cg = False
		self.UseBias = False
		self.NewtonLineSearch = False
		self.NewtonCholesky = False
		self.MaxIter = 200
		self.InitAlpha = False
		self.InitBias = 0
		self.CgStopType = 0
		self.CgStopParam = []
		self.CgStopIter = []
		self.UseOneClass = 0

class LapSVMData:

	def __init__(self, X, Y, options, K = None, L = None, Kv=None, Yv=None, sparse=True):
		self.X = X
		self.Y = Y

		self.options = options
		if K is None:
			self.K = CalcKernel(options, X)#ss.csr_matrix(X.dot(X.T))
		if L is None:
			self.L = gu.Laplacian(X, options)
			# if Lnormalized:
			#   if sparse:
			#       self.L = ss.diags([matrix_squeeze(self.K.sum(1))],[0]).tocsr() - self.K
			#   else:
			#       self.L = np.diag(matrix_squeeze(self.K.sum(1))) - self.K
			# else:
			#   if sparse:
			#       self.L = ss.diags([matrix_squeeze(self.K.sum(1))],[0]).tocsr() - self.K
			#   else:
			#       self.L = np.diag(matrix_squeeze(self.K.sum(1))) - self.K


		self.Kv = Kv
		self.Yv = Yv
		self.sparse = sparse

def cholupdate(R, x, sign):

	p = np.size(x)
	x = x.copy()
	x = x.T
	for k in range(p):
		if sign == '+':
			r = np.sqrt(R[k,k]**2 + x[k]**2)
		elif sign == '-':
			r = np.sqrt(R[k,k]**2 - x[k]**2)
		c = r/R[k,k]
		s = x[k]/R[k,k]
		R[k,k] = r
		if sign == '+':
			R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
		elif sign == '-':
			R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
		x[k+1:p] = c*x[k+1:p] - s*R[k, k+1:p]

	if np.isnan(R).any():
		raise Exception('Cholesky update led to non-PSD matrix.')
	return R

def CalcKernel(options, X1, X2=None):
	# % {calckernel} computes the Gram matrix of a specified kernel function.
	# % 
	# %      K = calckernel(options,X1)
	# %      K = calckernel(options,X1,X2)
	# %
	# %      options: a structure with the following fields
	# %               options.Kernel: 'linear' | 'poly' | 'rbf' 
	# %               options.KernelParam: specifies parameters for the kernel 
	# %                                    functions, i.e. degree for 'poly'; 
	# %                                    sigma for 'rbf'; can be ignored for 
	# %                                    linear kernel 
	# %      X1: N-by-D data matrix of N D-dimensional examples
	# %      X2: (it is optional) M-by-D data matrix of M D-dimensional examples
	# % 
	# %      K: N-by-N (if X2 is not specified) or M-by-N (if X2 is specified)
	# %         Gram matrix
	# %
	# % Author: Stefano Melacci (2009)
	# %         mela@dii.unisi.it
	# %         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

	kernel_type = options.Kernel
	kernel_param = options.KernelParam

	n1 = X1.shape[0]

	X2 = X1 if X2 is None else X2
	n2 = X2.shape[0]

	if kernel_type == 'linear':
		K = X2.dot(X1.T)
	elif kernel_type == 'poly':
		K = (X2.dot(X1.T))**kernel_param
	elif kernel_type == 'rbf':
		E = (np.tile(np.atleast_2d(np.sum(X1*X1,axis=1)),(n2,1)) + 
			 np.tile(np.atleast_2d(np.sum(X2*X2,axis=1)).T,(1,n1)) - 
			 2*X2.dot(X1.T))
		K = np.exp(-E/(2*(kernel_param**2)))
	else:
		raise Exception('Unknown kernel function.')

	return K


def trainLapSVM(data, options=LapSVMOptions()):
	data.X = np.array(data.X)
	data.Y = np.array(data.Y).squeeze()

	n,r = data.X.shape
	nn = np.sum(data.Y==1)

	if not hasattr(options,'Verbose'):
		options.Verbose = True
	if not hasattr(options,'Hinge'):
		options.Hinge = True # CHECK THIS
	if not hasattr(options,'Cg'):
		options.Cg = False # CHECK THIS
	if not hasattr(options,'MaxIter'):
		options.MaxIter = 200
	if not hasattr(options,'UseBias'):
		options.UseBias = False # CHECK THIS
	if not hasattr(options,'InitAlpha'):
		options.InitAlpha = False 
	if not hasattr(options,'InitBias'):
		options.InitBias = False # CHECK THIS
	if not hasattr(options,'NewtonLineSearch'):
		options.NewtonLineSearch = False
	if not hasattr(options,'NewtonCholesky'):
		options.NewtonCholesky = False
	if not hasattr(options,'CgStopType'):
		options.CgStopType = 0

	if options.CgStopType == 0: # none
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 0 # CHECK THIS
		if not hasattr(options,'CgStopIter'):
			options.CgStopIter = options.MaxIter + 1

	elif options.CgStopType == 1: # stability
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 0.015
		if not hasattr(options,'CgStopIter') or not options.CgStopIter:
			options.CgStopIter = int(round(np.sqrt(n)/2))

	elif options.CgStopType == 2: # validation
		v=len(data.Yv)
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 1 / v
		if not hasattr(options,'CgStopIter') or not options.CgStopIter:
			options.CgStopIter = int(round(np.sqrt(n)/2))

	elif options.CgStopType == 3: # stability and validation
		v=len(data.Yv)
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = [1 / v,0.015]
		if not hasattr(options,'CgStopIter'):
			options.CgStopIter = int(round(np.sqrt(n)/2))

	elif options.CgStopType == 4: # gradient norm
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 1e-08
		if not hasattr(options,'CgStopIter') or not options.CgStopIter:
			options.CgStopIter = 1

	elif options.CgStopType == 5: # preconditioned gradient norm
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 1e-08
		if not hasattr(options,'CgStopIter') or not options.CgStopIter:
			options.CgStopIter = 1

	elif options.CgStopType == 6: # mixed gradient norm
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 1e-08
		if not hasattr(options,'CgStopIter') or not options.CgStopIter:
			options.CgStopIter = 1

	elif options.CgStopType == 7: # relative objective function decrease
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 1e-06
		if not hasattr(options,'CgStopIter') or not options.CgStopIter:
			options.CgStopIter = 1

	elif options.CgStopType == -1: # debug
		if not hasattr(options,'CgStopParam') or not options.CgStopParam:
			options.CgStopParam = 0
		if not hasattr(options,'CgStopIter') or not options.CgStopIter:
			options.CgStopIter = options.MaxIter + 1

	else:
		raise Exception('Invalid CgStopType.')

	oc = 1 if nn == 0 else 0

	# initial alpha vector
	if options.InitAlpha and isinstance(options.InitAlpha, list):
		alpha = options.InitAlpha
	else:
		if not options.InitAlpha:
			alpha = np.array([])
		else:
			alpha = nr.randn(n,1)
			alpha = alpha / nlg.norm(alpha)

	# checking common error conditions
	if oc == 1 and not options.UseBias:
		raise Exception('One-Class SVM requires the UseBias option to be turned on.')
	if options.UseBias and options.LaplacianNormalize:
		raise Exception('The current implementation does not support a normalized Laplacian when the UseBias options is turned on.')

	# initial bias
	b = options.InitBias
	if options.Cg == 0:
		alpha,b,t,sec,lsiters = newton (data,options,alpha,b,oc)
		stats = []
	elif options.Cg == 1:
		alpha,b,t,sec,lsiters,stats = pcg(data,options,alpha,b,oc)
	else:
		raise Exception('Invalid solver specified in the field .Cg')

	svs = np.nonzero(alpha)[0]
	classifier = Classifier('lapsvmp', svs, alpha[svs], b, data.X[svs,:],
							 options, sec, t, lsiters, stats)
	return classifier

def newton(data, options=None, alpha=None, b=None, oc=None):

	t1 = time.time()
	n = len(data.Y)
	labeled = (data.Y != 0)
	l = np.sum(labeled)

	# Need to add this if it is not there.
	gamma_A = options.gamma_A
	gamma_I = options.gamma_I

	if not alpha:
		alpha = np.zeros(n)
		Kalpha = np.zeros(n)
	else:
		Kalpha = data.K.dot(alpha)

	t = 0
	lr = 0
	sv = np.zeros(n).astype(bool)

	lsiters= np.zeros(options.MaxIter)

	if gamma_I != 0:
		LK = data.L.dot(data.K)
	
	while True:
		if options.Hinge:
			sv_prev = sv.copy()
			hloss = ss.lil_matrix((n,1))
			hloss[labeled,:] = np.atleast_2d(1-data.Y[labeled]*(Kalpha[labeled] + b)).T
			hloss = hloss.tocsr()
			sv = matrix_squeeze((hloss > 0).todense())
			nsv = np.sum(sv)
		else:
			sv_prev = sv.copy()
			sv = labeled.copy()
			nsv = l

		if options.Verbose:
			if not options.Hinge:
				hloss = ss.lil_matrix((n,1))
				hloss[labeled,:] = np.atleast_2d(1-data.Y[labeled]*(Kalpha[labeled,:] + b)).T
				hloss = hloss.tocsr()
			if gamma_I != 0:
				obj=(gamma_A * alpha.T.dot(Kalpha) + 
					(hloss[sv,:].multiply(hloss[sv,:])).sum() + 
					gamma_I * Kalpha.T.dot(data.L.dot(Kalpha)) + oc * b) / 2
			else:
				obj=(gamma_A * alpha.T.dot(Kalpha) + 
					(hloss[sv,:].multiply(hloss[sv,:])).sum() + oc * b) / 2
			print('[t=%d] obj=%f nev=%d lr=%.4f\n'%(t,obj,nsv,lr))

		# goal conditions
		if t >= options.MaxIter:
			break
		if np.all(sv_prev == sv):
			break

		t = t + 1
		
		IsvK = ss.lil_matrix((n,n))
		IsvK[sv,:] = data.K[sv,:]
		IsvK = IsvK.tocsr()

		# computing new alphas
		onev = np.ones((1,n))

		if gamma_I == 0: # SVM (sparse solution)
			if options.UseBias:
				alpha_new = np.zeros(n)
				As = np.r_[ np.c_[0,onev[:nsv]], 
							np.c_[onev[:nsv].T,
								  gamma_A*np.eye(nsv)+matrix_squeeze(IsvK[sv,:][:,sv].todense())]]
				bs = np.squeeze(np.c_[oc/(2*gamma_A),data.Y[sv].T])
				alpha_b_new = nlg.solve(As, bs)
				alpha_new[sv] = alpha_b_new[1:]
				b_new = alpha_b_new[0]
			else:
				alpha_new = np.zeros(n)
				As = gamma_A*np.eye(nsv)+matrix_squeeze(IsvK[sv,:][:,sv].todense())
				alpha_new[sv] = nlg.solve(As,data.Y[sv])
				b_new = 0
		else: # LapSVM
			if options.NewtonCholesky: # inversion by factorization
				if t == 1:
					# compute the Cholesky factorization of the Hessian
					if options.UseBias:
						if data.sparse:
							sumKsv = np.atleast_2d(np.sum(matrix_squeeze(data.K[sv,:].todense()), axis=0))
							Mc = np.r_[np.c_[nsv , sumKsv], 
									   np.c_[ sumKsv.T, 
											  matrix_squeeze(data.K.dot(gamma_A*ss.eye(n) + IsvK + gamma_I*LK).todense())]]
						else:
							sumKsv = np.atleast_2d(matrix_squeezenp.np.sum(data.K[sv,:]), axis=0)
							Mc = np.r_[np.c_[nsv , sumKsv], 
									   np.c_[ sumKsv.T, 
											  matrix_squeeze(data.K.dot(gamma_A*np.eye(n) + IsvK.todense() + gamma_I*LK))]]

						hess = nlg.cholesky(Mc)
						if data.sparse:
							bs = matrix_squeeze(np.c_[np.sum(data.Y[sv])-oc/2, (data.K[:,sv].dot(data.Y[sv])).todense().T])
						else:
							bs = matrix_squeeze(np.c_[np.sum(data.Y[sv])-oc/2, data.K[:,sv].dot(data.Y[sv]).T])
						alpha_b_new = nlg.solve(hess,(nlg.solve(hess.T,bs)))
						alpha_new = alpha_b_new[1:]
						b_new = alpha_b_new[0]
					else: 
						if data.sparse:
							Mc = matrix_squeeze(data.K.dot(gamma_A*ss.eye(n) + IsvK + gamma_I*LK).todense())
						else:
							Mc = matrix_squeeze(data.K.dot(gamma_A*np.eye(n) + IsvK.todense() + gamma_I*LK))
						hess = nlg.cholesky(Mc)
						if data.sparse:
							bs = matrix_squeeze((data.K[:,sv].dot(data.Y[sv])).todense())
						else:
							bs = matrix_squeeze(data.K[:,sv].dot(data.Y[sv]).T)
						alpha_new = nlg.solve(hess,(nlg.solve(hess.T, bs)))
						b_new = 0
					LK = np.array([])
				
				else: # update the Cholesky factorization of the Hessian
					sv_diff = np.bitwise(np.bitwise_and(sv, sv_prev))
					sv_add = np.atleast_2d((np.bitwise_and(sv, sv_diff)).nonzero()[0]).T
					sv_rem = np.atleast_2d((np.bitwise_and(sv_prev, sv_diff)).nonzero()[0]).T
					if options.UseBias:
						if len(sv_add) > 0:
							for i in sv_add:
								if data.sparse:
									cvec = np.squeeze(np.c_[1, data.K[:,i].dense().T])
								else:
									cvec = np.squeeze(np.c_[1, data.K[:,i].T])
								hess = cholupdate(hess, cvec, '+')
						if len(sv_rem) > 0:
							for i in sv_rem:
								if data.sparse:
									cvec = np.squeeze(np.c_[1, data.K[:,i].dense().T])
								else:
									cvec = np.squeeze(np.c_[1, data.K[:,i].T])
								hess = cholupdate(hess, cvec, '-')

						if data.sparse:
							bs = matrix_squeeze(np.c_[np.sum(data.Y[sv])-oc/2, (data.K[:,sv].dot(data.Y[sv])).todense().T])
						else:
							bs = matrix_squeeze(np.c_[np.sum(data.Y[sv])-oc/2, data.K[:,sv].dot(data.Y[sv]).T])     
						alpha_b_new = nlg.solve(hess,(nlg.solve(hess.T,bs)))
						alpha_new = alpha_b_new[1:]
						b_new = alpha_b_new[0]
					
					else:
						if len(sv_add) > 0:
							for i in sv_add:
								hess = cholupdate(hess,data.K[:,i],'+')
						if len(sv_rem) > 0:
							for i in sv_rem.reshape(-1):
								hess = cholupdate(hess,data.K[:,i],'-')
						if data.sparse:
							bs = matrix_squeeze((data.K[:,sv].dot(data.Y[sv])).todense())
						else:
							bs = matrix_squeeze(data.K[:,sv].dot(data.Y[sv]).T)
						alpha_new = nlg.solve(hess,(nlg.solve(hess.T,bs)))
						b_new = 0
			else: # inversion without factorization
				IsvY = ss.lil_matrix((n,1))
				IsvY[sv] = np.atleast_2d(data.Y[sv]).T

				if options.UseBias:
					if data.sparse:
						As = np.r_[ np.c_[0,onev],
									np.c_[sv, (gamma_A*ss.eye(n)+IsvK+gamma_I*LK).todense()]]
					else:
						As = np.r_[ np.c_[0,onev],
									np.c_[sv, (gamma_A*np.eye(n)+IsvK.todense()+gamma_I*LK)]]
					bs = matrix_squeeze(np.c_[oc/(2*gamma_A),IsvY.todense().T])

					alpha_b_new = nlg.solve(As, bs)
					alpha_new = alpha_b_new[1:]
					b_new=alpha_b_new[0]
				else:
					alpha_new = nlg.solve(gamma_A*np.eye(n)+IsvK.todense()+gamma_I*LK.todense(),IsvY.todense())
					b_new = 0

		# step
		if options.NewtonLineSearch and (options.Hinge or np.any(alpha > 0)):
			step = alpha_new - alpha
			step_b = b_new - b

			lr,Kalpha,lsi,_ = linesearch(data,labeled,step,step_b,Kalpha,b,gamma_A,gamma_I,hinge=options.Hinge,oc=oc)
			alpha = alpha + lr * step
			b = b + lr * step_b
			lsiters[t] = lsi
		else:
			alpha = alpha_new.copy()
			b = b_new
			lr = 1
			lsiters[t] = 0
			Kalpha = data.K.dot(alpha)

	if options.Verbose:
		print('Done with Newton\'s method.')
	lsiters = lsiters[0:t]
	sec = time.time() - t1
	return alpha,b,t,sec,lsiters

def pcg(data=None, options=None,alpha=None,b=None,oc=None):
# {pcg} trains the classifier using preconditioned conjugate gradient.
	t1 = time.time()

	n = len(data.Y)
	labeled = (data.Y != 0)
	unlabeled = np.bitwise_not(labeled)
	l = np.sum(labeled)
	u = n - l

	if data.Yv is not None:
		v = len(data.Yv)
	# Need to add this if it is not there.
	gamma_A=options.gamma_A
	gamma_I=options.gamma_I

	if len(alpha) == 0:
		alpha = np.zeros(n)
		Kalpha = np.zeros(n)
		if gamma_I != 0:
			LKalpha = np.zeros(n)
		else:
			LKalpha = np.array([])
		go = data.Y - b * labeled
		obj0 = (np.sum((1-data.Y[labeled]*b* options.UseBias)**2)+oc*b)/2
		if options.UseBias:
			go_b = np.sum(data.Y[labeled]-b)-oc / 2
		else:
			go_b = 0
			b = 0
	else:
		Kalpha = data.K.dot(alpha)
		if gamma_I != 0:
			LKalpha = data.L.dot(Kalpha)
		else:
			LKalpha = np.array([])
		out = np.zeros(n)
		out[labeled] = Kalpha[labeled] + b

		if options.Hinge:
			sv = np.zeros(n).astype(bool)
			sv[labeled]= ((data.Y[labeled].dot(out[labeled])) < 1)
		else:
			sv = labeled.copy()
		go =- gamma_A*alpha
		if gamma_I != 0:
			go = go - gamma_I*LKalpha
		obj0 = (Kalpha.T.dot(-go) + np.sum((out[sv]- data.Y[sv])**2) + oc*b) / 2
		go[sv] = go[sv]-(out[sv] - data.Y[sv])
		if options.UseBias:
			go_b = np.sum(data.Y[sv]-Kalpha[sv]-b)-oc/2
		else:
			go_b = 0
			b = 0

	d = go.copy() # initial search direction
	d_b = go_b.copy()
	Kgo = data.K.dot(go)
	Kstep = Kgo.copy()

	t = 0
	stats = np.array([])
	
	if options.CgStopType == 4:
		ng0 = np.sqrt(np.sum(Kgo**2) + go_b**2)
	elif options.CgStopType == 5:
		npg0 = np.sqrt(np.sum(go**2) + go_b**2)
	elif options.CgStopType == 6:
		ngm0 = np.sqrt(Kgo.dot(go) + go_b**2)
	elif options.CgStopType == -1:
		stats = np.zeros((options.MaxIter + 1,5 + n + 1))
		ng0 = np.sqrt(np.sum(Kgo**2) + go_b**2)
		ngp0= np.sqrt(np.sum(go**2) + go_b**2)
		ngm0= np.sqrt(np.sum(Kgo.dot(go) + go_b ** 2))
		# What?
		stats[t+1,:] = np.c_[t,obj0/obj0,ng0/ng0,ngp0/ngp0,ngm0/ ngm0,alpha,b]


	lsiters = np.zeros(options.MaxIter)
	valerr_prev = 1
	obj_prev = obj0
	yfx_unlabeled_prev = np.zeros(u).astype(bool)	

	while True:
		t = t + 1

		# goal condition: maximum number of iterations
		if t > options.MaxIter:
			t = t - 1
			break

		# do an exact line search 
		lr,Kalpha,lsi,LKalpha = linesearch(data,labeled,d,d_b,Kalpha,b,gamma_A,gamma_I,Kstep,LKalpha,options.Hinge,oc)

		# goal condition: converged to optimal solution
		if lr == 0:
			t = t - 1
			break
		
		alpha = alpha + lr * d
		b = b + lr * d_b
		lsiters[t] = lsi

		# compute new precgradient and objective 
		out = np.zeros(n)
		out[labeled] = Kalpha[labeled] + b

		if options.Hinge:
			sv = np.zeros(n).astype(bool)
			sv[labeled] = (data.Y[labeled]*out[labeled] < 1)
		else:
			sv = labeled.copy()

		g = gamma_A * alpha
		if gamma_I != 0:
			g=g + gamma_I * LKalpha

		if options.Verbose or options.CgStopType == 7 or options.CgStopType == -1:
			obj=(Kalpha.dot(g) + np.sum((out[sv] - data.Y[sv])**2) + oc*b) / 2
			if options.Verbose:
				print('[t=%d] obj=%f nev=%d lr=%.4f\n'%(t,obj,np.sum(sv),lr))
		
		g[sv] = g[sv] + (out[sv] - data.Y[sv])
		
		if options.UseBias:
			g_b = np.sum(out[sv]-data.Y[sv]) + oc / 2
		else:
			g_b = 0
		
		gn = -g
		gn_b = -g_b

		# goal condition: data based
		if (t-1)%options.CgStopIter == options.CgStopIter-1:
			if options.CgStopType == 1: # stability
				yfx_unlabeled = np.sign(Kalpha[unlabeled] + b)
				diff_unlabeled = 1 - np.sum(yfx_unlabeled == yfx_unlabeled_prev)/u
				if diff_unlabeled < options.CgStopParam:
					break
				yfx_unlabeled_prev=yfx_unlabeled.copy()

			elif options.CgStopType == 2: # validation
				valerr = 1 - np.sum(np.sign(data.Kv.dot(alpha)+b)==data.Yv)/v
				if (valerr > (valerr_prev - options.CgStopParam)):
					break
				valerr_prev = valerr.copy()

			elif options.CgStopType == 3: # stability and validation
				valerr = 1 - np.sum(np.sign(data.Kv.dot(alpha) + b) == data.Yv) / v
				yfx_unlabeled = np.sign(Kalpha[unlabeled] + b)
				diff_unlabeled = 1 - np.sum(yfx_unlabeled == yfx_unlabeled_prev) / u
				if (valerr > (valerr_prev - options.CgStopParam[0])) and diff_unlabeled < options.CgStopParam[1]:
					break
				valerr_prev = valerr.copy()
				yfx_unlabeled_prev = yfx_unlabeled.copy()

			elif options.CgStopType == 4: # gradient norm
				ng = np.sqrt(np.sum(gn**2) + gn_b**2)
				if ng/ng0 < options.CgStopParam:
					break

			elif options.CgStopType == 5: # preconditioned gradient norm
				ngp = np.sqrt(np.sum(go** 2) + go_b ** 2)
				if ngp / ngp0 < options.CgStopParam:
					break

			elif options.CgStopType == 6: # mixed gradient norm
				ngm = np.sqrt(Kgo.T.dot(go) + go_b ** 2)
				if ngm / ngm0 < options.CgStopParam:
					break

			elif options.CgStopType == 7: # relative objective function decrease
				if (obj_prev - obj) < options.CgStopParam:
					break
				obj_prev = obj.copy()
		
		Kgn = data.K.dot(gn) # multiply by the preconditioner
		
		# debug
		if options.CgStopType == - 1:
			ng = np.sqrt(np.sum(Kgn**2) + gn_b ** 2)
			ngp = np.sqrt(np.sum(gn ** 2) + gn_b ** 2)
			ngm = np.sqrt(Kgn.T.dot(gn) + gn_b ** 2)
			stats[t+1,:]=np.c_[t,obj/obj0,ng/ng0,ngp/ngp0,ngm/ngm0,alpha,b]

		# Polack-Ribiere update with automatic restart
		# IPython.embed()
		be = max(0,(Kgn.T.dot(gn-go) + gn_b*(gn_b-go_b))/(Kgo.T.dot(go) + go_b**2))
		d = be*d + gn
		d_b = be*d_b + gn_b
		Kstep = be * Kstep + Kgn
		go = gn.copy()
		go_b = gn_b.copy()
		Kgo = Kgn.copy()

	sec = time.time() - t1

	if options.Verbose:
		print('Done with PCG.')
	lsiters = lsiters[0:t]
	if len(stats) > 0:
		stats=stats[0:t,:]

	return alpha,b,t,sec,lsiters,stats

def linesearch(data,labeled=None,step=None,step_b=None,Kalpha=None,b=None,gamma_A=None,gamma_I=None,Kstep=None,LKalpha=None,hinge=None,oc=None):
	# {linesearch} does a line search in direction step.
	
	act = (step != 0) # the set of points for which alpha change (active)
	if len(Kstep) == 0:
		Kstep = data.K[:,act].dot(step[act])

	# precomputations
	stepKstep = step[act].T.dot(Kstep[act])
	stepKalpha = step[act].T.dot(Kalpha[act])
	if gamma_I != 0:
		KstepL = ss.csr_matrix(Kstep).dot(data.L)
		KstepLKalpha = KstepL.dot(Kalpha)
		KstepLKstep = KstepL.dot(Kstep)

	out = Kalpha[labeled] + b
	outstep = Kstep[labeled] + step_b
	out_minus_Y = out - data.Y[labeled]

	# breakpoints
	if hinge:
		sv = ((1 - out*data.Y[labeled]) > 0)
		deltas = -out_minus_Y / outstep
		deltas[deltas < 0] = 0
		deltas_map = np.argsort(deltas)
		deltas = deltas[deltas_map]
		lab = len(deltas)
		i = np.nonzero(deltas > 0)[0][:1]
		lsi = 1
	else:
		sv = np.ones(len(out)).astype(bool)
		lsi = 1

	# intercepts
	if gamma_I != 0:
		left = (outstep[sv].T.dot(out_minus_Y[sv]) + gamma_A*stepKalpha + 
				 gamma_I * KstepLKalpha + (oc/2)*step_b)
		right = (outstep[sv].T.dot(outstep[sv]) + gamma_A*stepKstep + 
				 gamma_I * KstepLKstep)
	else:
		left = outstep[sv].T.dot(out_minus_Y[sv]) + gamma_A*stepKalpha + (oc/2)*step_b
		right = outstep[sv].T.dot(outstep[sv]) + gamma_A*stepKstep

	# first minimum
	zcross = -left/right
	if right <= 0:
		zcross = 0

	if hinge and len(i) > 0:
		i = i[0]
		if sv[deltas_map[i]]:
			if 0 <= zcross and zcross < deltas[i]:
				not_got_it = 0
			else:
				not_got_it=1
		else:
			if 0 <= zcross and zcross <= deltas[i]:
				not_got_it = 0
			else:
				not_got_it = 1

		while not_got_it:
			# updating support vectors
			j = -2*sv[deltas_map[i]] + 1
			sv[deltas_map[i]] = not sv[deltas_map[i]]

			# updating intercepts
			left = left + j*outstep[deltas_map[i]]*(out_minus_Y[deltas_map[i]])
			right = right + j*outstep[deltas_map[i]]*(outstep[deltas_map[i]])

			# computing minimum
			zcross = -left/right

			# goal conditions
			if i == lab:
				break
			if sv[deltas_map[i]]:
				if sv[deltas_map[i+1]]:
					if deltas[i] < zcross and zcross < deltas[i+1]:
						not_got_it = 0
				else:
					if deltas[i] < zcross and zcross <= deltas[i+1]:
						not_got_it = 0
			else:
				if sv[deltas_map[i+1]] is True:
					if deltas[i] <= zcross and zcross < deltas[i+1]:
						not_got_it = 0
				else:
					if deltas[i] <= zcross and zcross <= deltas[i+1]:
						not_got_it = 0
			i = i + 1
			lsi = lsi + 1

	lr = np.squeeze(zcross)

	if lr < 0: # converged
		lr = 0
		return lr,Kalpha,lsi,LKalpha

	Kalpha = Kalpha + lr * Kstep
	if gamma_I != 0: # what
		LKalpha = LKalpha + lr*matrix_squeeze(KstepL.todense())
	return lr,Kalpha,lsi,LKalpha