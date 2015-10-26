# -*- coding: utf-8 -*-
import numpy as np
import cmath
from numpy.polynomial.polynomial import polyval, polyadd, polysub, polydiv, polymul, polymulx, polyroots
from random import shuffle
from scipy.stats import bernoulli
from sklearn.utils.extmath import randomized_svd
from cvxopt import matrix, spmatrix, solvers

#####################################################################
#### LowRankBiLinear #####
# Method due to Liu et al. (2015) Low-Rank Similarity Metric Learning in High Dimensions

def Tfunc(M,theta):
   n,m = M.shape
   out = np.zeros((n,m))
   for i in xrange(n):
     for j in xrange(m):
        if M[i,j]>theta: out[i,j]=M[i,j]-theta
        elif M[i,j]<0.0: out[i,j]=M[i,j]
   return(out)

def nearPSD(A,epsilon=0):
   n = A.shape[0]
   if n==1: return(np.maximum(A,epsilon))
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

def LowRankBiLinear(m,X,Y,eps,rho,tau,T,tol=1e-6,epsilon=0.0):
    # m - dimension of similarity function
    # X - n x d data matrix. n << d. n observations, d dimensions
    # Y - n x 1 label vector
    # eps - ?
    # rho - ?
    # alpha - regularization strength
    # tau - ?
    # T - iteration limit
    n,d = X.shape
    Ym = -np.matrix(np.ones((n,n)))
    Ytil = eps*np.matrix(np.ones((n,n)))
    for y in np.unique(Y):
      y_vec = Y==y
      Ym = Ym + 2*np.outer(y_vec,y_vec)
      Ytil = Ytil + (1-eps)*np.outer(y_vec,y_vec)
    X = X.T 
    U, E, V = randomized_svd(X,n_components=m,n_iter=5,random_state=None)
    U = np.asmatrix(U)
    E2 = np.matrix(np.diag(E**2))
    Xtil = U.T*X
    W = np.matrix(np.identity(m))
    I = np.matrix(np.identity(m))
    L = np.matrix(np.zeros((n,n)))
    S = Xtil.T*Xtil
    for k in xrange(T):
      Z = Tfunc(Ytil-np.multiply(Ym,S)-L,1.0/rho)
      G = alpha/rho*I+Xtil*np.multiply(Ym,Z-Ytil+L)*Xtil.T + E2*W*E2
      W = nearPSD(W-tau*G,epsilon)
      S = Xtil.T*W*Xtil
      Delta = Z-Ytil+np.multiply(Ym,S)
      L = L + Delta
      if np.sum(np.multiply(Delta,Delta))/n**2 <= tol:
        break
    E,H = np.linalg.eig(W)
    E = np.maximum(E,epsilon)
    out = U*H*np.diag(np.sqrt(E))
    return(out)

#### LowRankBiLinear Example #####
#m = 2
#X = np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
#Y = np.array([1, 1, 1, 0, 0, 0])
#eps = 0.1
#rho = 1
#alpha = 0.5
#tau = 0.01
#T = 10
#tol = 1e-6
#L = LowRankBiLinear(m,X,Y,eps,rho,tau,T)

#####################################################################
#### HingeLossQuadratic #####
# Method due to Guo & Ying (2014) Guaranteed Classification via Regularized Similarity Learning
# except here we use the square of the frobenius norm for regularization
# optimized using quadratic programming

def HingeLossQuadratic(X,Y,r,alpha,R = None, Symmetric = False):
  # X - data matrix, each row is an observation
  # Y - label vector {-1,1}
  # r - margin - to be chosen with CV
  # alpha - regularization strength - to be chosen with CV
  # R - numpy logical vector indicating which points are reasonable
  # Symmetric - bool, if True we will find a symmetric matrix
  M,N = X.shape # M - number of observations, N - dimension of each
  N2 = N*N
  if R==None:
    R = np.array([True]*M)
  NR = np.sum(R)
  P = spmatrix(alpha,range(N2),range(N2),(N2+M,N2+M),tc='d')
  q = spmatrix(1.0/M,range(N2,N2+M),[0]*M,tc='d')
  q = matrix(q)
  h = spmatrix(1.0,range(M,2*M),[0]*M,tc='d')
  h = matrix(h)
  Xy = np.diag(Y)*X
  d1 = Xy[R,].sum(axis=0) # column sums for reasonable observations
  GI = [0]*(M*N2+2*M)
  GJ = [0]*(M*N2+2*M)
  GV = [0]*(M*N2+2*M)
  for i in xrange(M):
    for j in xrange(N2):
      k = np.floor(j/N) # This indicates that x[:N2] will be the elements of A in row-major order
      l = np.mod(j,N)
      idx = i*N2+j
      GI[idx] = i+M
      GJ[idx] = j
      GV[idx] = -Xy[i,k]*d1[0,l]/(NR*r)
    idx = i+M*N2
    GI[idx] = i
    GJ[idx] = i+N2
    GV[idx] = -1.0
    idx = i+M*N2+M
    GI[idx] = i+M
    GJ[idx] = i+N2
    GV[idx] = -1.0
  G = spmatrix(GV,GI,GJ,tc='d')
  if Symmetric:
    Nc2 = N*(N-1)/2
    A = spmatrix(0,[],[],(Nc2,N2+M),tc='d')
    b = matrix(0,(N*(N-1)/2,1),tc='d')
    r = 0
    for i in xrange(N):
      for j in xrange(i+1,N):
        A[r,i*N+j] = 1
        A[r,j*N+i] = -1
        r = r+1
  if Symmetric: sol=solvers.qp(P, q, G, h, A, b)
  else: sol=solvers.qp(P, q, G, h)
  Asim = np.matrix(sol['x'][:N2]).reshape((N,N)) # expects x in row-major order
  return(Asim)

### HingeLossQuadratic Example #####
#X = np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
#Y = np.array([1, 1, 1, -1, -1, -1])
#A = HingeLossQuadratic(X,Y,1e-2,1e-8,Symmetric=True)

#####################################################################
#### OASIS #####
# Method due to Chechik et al. (2010) Large Scale Online Learning of Image Similarity Through Ranking

def OASIS(X,Y,C,itmax=10,batch_size=None,loss_tol=1e-3):
  # X - data matrix, each row is an observation
  # Y - label vector (any type)
  # C - aggressiveness parameter, balances trade-off between fidelity to previous solution and correction for current loss
  # batch_size - number of random samples for which to calculate expected loss, default is the number of observations
  # loss_tol - stopping criterion, if estimate of expected loss falls below loss_tol we break
  M,N = X.shape # M - number of observations, N - dimension of each
  if batch_size==None: batch_size = M
  Class = {}
  nC = {}
  for i in xrange(len(Y)):
    if Y[i] not in Class:
      Class[Y[i]] = []
      nC[Y[i]] = 0
    Class[Y[i]].append(i)
    nC[Y[i]] += 1
  nY = len(Y)
  W = np.identity(N)
  for k in xrange(itmax):
    totloss = 0.0
    totloss2 = 0.0
    for i in xrange(batch_size):
      # randomly sample triplet
      idx = np.random.randint(0,nY)
      c = Y[idx]
      r_ref = X[idx,]
      pos_idx = Class[c][np.random.randint(0,nC[c])]
      neg_idx = np.random.randint(0,nY-nC[c])
      for neg_c in Class:
        if neg_c==c: continue
        if pos_idx>=nC[neg_c]: pos_idx -= nC[neg_c]
        else: 
          neg_idx = Class[neg_c][neg_idx]
          break
      r_pos = X[pos_idx,]
      r_neg = X[neg_idx,]
      s1 = r_ref*W*r_pos.T
      s2 = r_ref*W*r_neg.T
      loss = max(0,1-s1[0,0]+s2[0,0])
      totloss = totloss+loss
      totloss2= totloss2+loss**2
      V = r_ref.T*(r_pos-r_neg)
      norm = np.linalg.norm(V,'fro')**2
      tao = min(C,loss/norm)
      W = W + tao*V
    if totloss < batch_size*loss_tol:
      break
  loss = totloss/float(batch_size)
  loss_sig = np.sqrt((totloss2-2*loss*totloss+batch_size*loss**2)/float(batch_size-1))
  lossCI = 1.96*loss_sig/np.sqrt(batch_size)
  if totloss < batch_size*loss_tol:
    print "Stopping criterion met in "+str(k)+" iterations. Expected loss: "+str(loss)+" +/-"+str(lossCI)+" (at 95% confidence)"
  else:
    print "Maximum number of iterations ("+str(itmax)+") exceeded. Expected loss: "+str(loss)+" +/-"+str(lossCI)+" (at 95% confidence)"
  return(W)

#### OASIS Example #####
#X = np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
#Y = np.array([1, 1, 1, -1, -1, -1])
#A = OASIS(X,Y,1.0,100)

#####################################################################
#### OASIS_SIM #####
# Symmetric modification of OASIS by Kyle Miller

def OASIS_SIM(m,X,Y,C,itmax=10,batch_size=None,loss_tol=1e-3,epsilon=1e-10):
  # m - projection dimension
  # X - data matrix, each row is an observation
  # Y - label vector (any type)
  # C - aggressiveness parameter, balances trade-off between fidelity to previous solution and correction for current loss
  # batch_size - number of random samples for which to calculate expected loss, default is the number of observations
  # loss_tol - stopping criterion, if estimate of expected loss falls below loss_tol we break
  # epsilon - size of zero. If absolute value of a number is less than epsilon, a number is considered zero
  C_limit = C
  M,N = X.shape # M - number of observations, N - dimension of each
  if batch_size==None: batch_size = M
  Class = {}
  nC = {}
  for i in xrange(len(Y)):
    if Y[i] not in Class:
      Class[Y[i]] = []
      nC[Y[i]] = 0
    Class[Y[i]].append(i)
    nC[Y[i]] += 1
  nY = len(Y)
  L = np.asmatrix(np.ones((m,N)))/(m**2*N**2)
  for k in xrange(itmax):
    totloss = 0.0
    totloss2 = 0.0
    for i in xrange(batch_size):
      # randomly sample triplet
      idx = np.random.randint(0,nY)
      c = Y[idx]
      r_ref = X[idx,]
      pos_idx = Class[c][np.random.randint(0,nC[c])]
      neg_idx = np.random.randint(0,nY-nC[c])
      for neg_c in Class:
        if neg_c==c: continue
        if pos_idx>=nC[neg_c]: pos_idx -= nC[neg_c]
        else: 
          neg_idx = Class[neg_c][neg_idx]
          break
      r_pos = X[pos_idx,]
      r_neg = X[neg_idx,]
      a = r_ref
      b = r_pos-r_neg
      ab = np.sum(a*b.T)
      aa = np.sum(a*a.T)
      bb = np.sum(b*b.T)
      if aa==0 or bb==0: continue
      La = L*a.T
      Lb = L*b.T
      aLLb = np.sum(La.T*Lb)
      aLLa = np.sum(La.T*La)
      bLLb = np.sum(Lb.T*Lb)
      if 1-aLLb<=0: continue # if non-positive loss, don't do anything
      totloss = totloss+(1-aLLb)
      totloss2 = totloss2+(1-aLLb)**2
      # f function coefficients
      A42 = 0.5*(aa*bb-ab**2)*(aa*bLLb+bb*aLLa-2*ab*aLLb)
      A32 = 2*(aa*bb-ab**2)*aLLb
      A22 = 0.5*(aa*bLLb+bb*aLLa+2*ab*aLLb)
      # q polynomial coefficients
      a4=(aa*bb-ab**2)**2
      a3=4.0*ab*(aa*bb-ab**2)
      a2=ab*(aLLa*bb+aa*bLLb)+((1.0-aLLb)-3.0)*(aa*bb-ab**2)+2.0*ab**2*((1.0-aLLb)+1.0)
      a1=-aLLa*bb+2.0*aLLb*ab-aa*bLLb-4.0*ab
      a0=1.0-aLLb
      #1. depress via substitution tau = y - a3/(4*a4) => y^4+ A*y^2+ B*y +C = 0
      A = (-3.0*a3**2/(8.0*a4)+a2)/a4
      B = (a3**3/(8.0*a4**2)-a2*a3/(2.0*a4)+a1)/a4
      C = (-3.0*a3**4/(256.0*a4**3)+a3**2*a2/(16.0*a4**2)-a3*a1/(4.0*a4)+a0)/a4
      #2. solve resolvant cubic for a root. z^3 + c2*z^2 + c1*z + c0 = 0
      c2 = 3.0*cmath.sqrt(C)-0.5*A
      c1 = 2.0*C-A*cmath.sqrt(C)
      c0 = -B**2/8.0
      Delta0 = c2**2-3.0*c1
      Delta1 = 2.0*c2**3-9.0*c2*c1+27.0*c0
      Delta = cmath.sqrt(-27.0*(18.0*c2*c1*c0-4.0*c2**3*c0+c2**2*c1**2-4.0*c1**3-27.0*c0**2))
      C2 = (0.5*(Delta1+Delta))**(1.0/3.0)
      z = -1.0/3.0*(c2+C2+Delta0/C2)
      #3. Complete the square and solve for first two roots
      D = cmath.sqrt(2.0*cmath.sqrt(C)-A+2.0*z)
      E = cmath.sqrt(2.0*cmath.sqrt(C)*z+z*z)
      b1 = a3/(2.0*a4)+D
      c1 = a3**2/(16.0*a4**2)+D*a3/(4.0*a4)+E+z+cmath.sqrt(C)
      #3a. factor quartic polynomial into two quadratics
      factor2 = polydiv((a0,a1,a2,a3,a4),(c1,b1,1))[0]
      b2 = factor2[1]/factor2[2]
      c2 = factor2[0]/factor2[2]
      #4. Examine roots for each quadratic, find real positive ones.
      tau = C_limit
      d = (1-C_limit*ab)**2-aa*bb*C_limit**2
      if abs(d)>=epsilon:
         q = a4*C_limit**4+a3*C_limit**3+a2*C_limit**2+a1*C_limit+a0
         obj = ((A42*C_limit**2+A32*C_limit+A22)*C_limit**2+C_limit*max(0,q))/(d**2)
      else: # we have a problem, d=0 at C_limit
         obj = float("inf")
      optimal = (C_limit,obj)
      roots = [0.5*(-b1+cmath.sqrt(b1*b1-4.0*c1)),0.5*(-b1-cmath.sqrt(b1*b1-4.0*c1)),0.5*(-b2+cmath.sqrt(b2*b2-4.0*c2)),0.5*(-b2-cmath.sqrt(b2*b2-4.0*c2))]
      roots = [r.real for r in roots if abs(r.imag)<epsilon and r.real>0 and r.real<C_limit]
      for r in roots:
        d = (1-r*ab)**2-aa*bb*r**2
        obj = (A42*r**2+A32*r+A22)*r**2/(d**2)        
        if obj<optimal[1]:
          optimal = (r,obj)
      if optimal[1]<float("inf"):
        tau = optimal[0]
        d = (1-tau*ab)**2-aa*bb*tau**2
        updateL = tau/d*((1.0-tau*ab)*(La*b+Lb*a)+(tau*bb*La)*a+(tau*aa*Lb)*b)
        L = L + updateL
    if totloss < batch_size*loss_tol:
      break
  loss = totloss/float(batch_size)
  loss_sig = np.sqrt((totloss2-2*loss*totloss+batch_size*loss**2)/float(batch_size-1))
  lossCI = 1.96*loss_sig/np.sqrt(batch_size)
  if totloss < batch_size*loss_tol:
    print "Stopping criterion met in "+str(k)+" iterations. Expected loss: "+str(loss)+" +/-"+str(lossCI)+" (at 95% confidence)"
  else:
    print "Maximum number of iterations ("+str(itmax)+") exceeded. Expected loss: "+str(loss)+" +/-"+str(lossCI)+" (at 95% confidence)"
  return(L)

#### OASIS_SIM Example #####
#X = np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
#Y = np.array([1, 1, 1, -1, -1, -1])
#A = OASIS_SIM(2,X,Y,1.0,100)

#####################################################################
#### JSL #####
# Method due to Nicolae, et al. (2015) Joint Semi-supervised Similarity Learning for Linear Classification
# with extension to multiple subproblems by Kyle Miller

def JSL_subroutine_alpha(A,c,G,h,X,Y,LandMarks,N):
  K = len(LandMarks)
  row_grp = 0
  totD = 0
  for k in xrange(K):
    Y_k = Y[k]
    X_k = X[k]
    L = LandMarks[k]
    D = len(L)
    M = len(Y_k)
    for i in xrange(M):
      row_idx = row_grp + i
      y_ki = Y_k[i]
      x = X_k[i]
      for j in xrange(D):
        G[row_idx,totD+j] = -y_ki* ( x*A*L[j].T ) # bilinear similarity
    row_grp += M
    totD += D
  sol = solvers.lp(c,G,h)
  if sol['status']!='optimal':
    print 'WARNING! Optimization alpha step returned status: '+sol['status']
  return sol['x'][:totD],sol['primal objective']

def JSL_subroutine_A(alpha,c,G,h,X,Y,LandMarks,N,Diagonal,Symmetric,A,b):
  K = len(LandMarks)
  N2 = N*N
  landmarks = {}
  for k in xrange(K):
    for j in xrange(len(LandMarks[k])):
      if k in landmarks: landmarks[k] += LandMarks[k][j]*alpha[k*K+j,0]
      else: landmarks[k] = LandMarks[k][j]*alpha[k*K+j,0]
  row_grp = 0
  if not Diagonal:
    for k in xrange(K):
      Y_k = Y[k]
      X_k = X[k]
      L = landmarks[k]
      M = len(Y_k)
      for i in xrange(M):
        row_idx = row_grp + i
        y_ki = Y_k[i]
        x = y_ki*X_k[i]
        for n in xrange(N):
          for m in xrange(N):
            G[row_idx,n*N+m] = -x[0,n]*L[0,m]
      row_grp += M
    if not Symmetric: sol = solvers.lp(c,G,h)
    else: sol = solvers.lp(c,G,h,A,b)
    if sol['status']!='optimal':
      print 'WARNING! Optimization A step returned status: '+sol['status']
    return np.matrix(sol['x'][:N2]).reshape((N,N)),sol['primal objective'] # expects x in row-major order
  else:
    for k in xrange(K):
      Y_k = Y[k]
      X_k = X[k]
      L = landmarks[k]
      M = len(Y_k)
      for i in xrange(M):
        row_idx = row_grp + i
        y_ki = Y_k[i]
        x = y_ki*X_k[i]
        for n in xrange(N):
          G[row_idx,n] = -x[0,n]*L[0,n]
      row_grp += M
    if not Symmetric: sol = solvers.lp(c,G,h)
    else: sol = solvers.lp(c,G,h,A,b)
    if sol['status']!='optimal':
      print 'WARNING! Optimization A step returned status: '+sol['status']
    return np.asmatrix(np.diag([sol['x'][i,0] for i in xrange(N)])),sol['primal objective']
  
def JSL(X,Y,LandMarks,w=None,lambda_val=1.0,gamma=None,R=None,Diagonal=False,Symmetric=False,MaxIts=10,Tol=1e-8):
  N = X[0].shape[1]
  N2 = N*N
  K = len(LandMarks)
  totD = sum([len(L) for L in LandMarks]) # total number of parameters for linear classifiers
  totM = sum([len(Y_k) for Y_k in Y]) # total number of labeled examples
  if Diagonal: Symmetric=False
  if R==None: R = np.asmatrix(np.identity(N))
  if w==None: w = [1.0]*K
  if gamma==None: gamma = [1.0]*K
  A = R
  w = [w[i]/float(len(Y[i])) for i in xrange(len(w))]
  alpha = matrix(np.repeat([0]*K,[len(Y_k) for Y_k in Y]),tc='d')
  row_grp = 0
  if not Diagonal: 
    c_A = [0]*N2+[0]*N2 + np.repeat(w,[len(Y_k) for Y_k in Y]).tolist()+[lambda_val]
    h_A = [-1]*totM+[0]*totM+ R.flatten().tolist()[0] + (-R).flatten().tolist()[0] + [0]*N + [0]
    G_A = spmatrix([],[],[],(2*totM+2*N2+N+1,2*N2+totM+1),tc='d')
    for k in xrange(K):
      M = len(Y[k])
      for i in xrange(M):
        G_A[row_grp+i,2*N2+row_grp+i] = -1
        G_A[totM+row_grp+i,2*N2+row_grp+i] = -1
      row_grp += M
    for i in xrange(N2):
      G_A[2*totM+i,i] = 1
      G_A[2*totM+i,N2+i] = -1
      G_A[2*totM+N2+i,i] = -1
      G_A[2*totM+N2+i,N2+i] = -1
    for i in xrange(N):
      G_A[2*totM+2*N2+i,2*N2+totM] = -1
      for j in xrange(N):
        G_A[2*totM+2*N2+i,N2+i*N+j] = 1
    G_A[2*totM+2*N2+N,2*N2+totM] = -1
  else:
    c_A = [0]*N+[0]*N + np.repeat(w,[len(Y_k) for Y_k in Y]).tolist()+[lambda_val]
    h_A = [-1]*totM+[0]*totM+ np.diag(R).tolist() + np.diag(-R).tolist() + [0]*N + [0]
    G_A = spmatrix([],[],[],(2*totM+2*N+N+1,2*N+totM+1),tc='d')
    for k in xrange(K):
      M = len(Y[k])
      for i in xrange(M):
        G_A[row_grp+i,2*N+row_grp+i] = -1
        G_A[totM+row_grp+i,2*N+row_grp+i] = -1
      row_grp += M
    for i in xrange(N):
      G_A[2*totM+i,i] = 1
      G_A[2*totM+i,N+i] = -1
      G_A[2*totM+N+i,i] = -1
      G_A[2*totM+N+i,N+i] = -1
    for i in xrange(N):
      G_A[2*totM+2*N+i,2*N+totM] = -1
      G_A[2*totM+2*N+i,N+i] = 1
    G_A[2*totM+2*N+N,2*N+totM] = -1
  c_A = matrix(c_A,tc='d')
  h_A = matrix(h_A,tc='d')
  c_alpha = [0]*totD+[0]*totD+np.repeat(w,[len(Y_k) for Y_k in Y]).tolist()
  c_alpha = matrix(c_alpha,tc='d')
  h_alpha = [-1]*totM+[0]*totM+[0]*totD+[0]*totD+[1.0/g for g in gamma]
  h_alpha = matrix(h_alpha,tc='d')
  G_alpha = spmatrix([],[],[],(2*totM+2*totD+K,2*totD+totM),tc='d')
  row_grp = 0
  rowD = 0
  for k in xrange(K):
    M = len(Y[k])
    D = len(LandMarks[k])
    for i in xrange(M):
      G_alpha[row_grp+i,2*totD+row_grp+i] = -1
      G_alpha[totM+row_grp+i,2*totD+row_grp+i] = -1
    row_grp += M
    for j in xrange(D):
      G_alpha[2*totM+rowD+j,rowD+j] = 1
      G_alpha[2*totM+rowD+j,totD+rowD+j] = -1
      G_alpha[2*totM+totD+rowD+j,rowD+j] = -1
      G_alpha[2*totM+totD+rowD+j,totD+rowD+j] = -1
      G_alpha[2*totM+2*totD+k,totD+rowD+j] = 1
    rowD += D
  A_A = None
  b_A = None
  if not Diagonal and Symmetric:
    Nc2 = N*(N-1)/2
    A_A = spmatrix([],[],[],(Nc2,G_A.size[1]),tc='d')
    b_A = matrix(0,(Nc2,1),tc='d')
    r = 0
    for i in xrange(N):
      for j in xrange(i+1,N):
        A_A[r,i*N+j] = 1
        A_A[r,j*N+i] = -1
        r = r+1
  v0 = 0
  for i in xrange(MaxIts):
    A,v_A = JSL_subroutine_A(alpha,c_A,G_A,h_A,X,Y,LandMarks,N,Diagonal,Symmetric,A_A,b_A)
    alpha,v_alpha = JSL_subroutine_alpha(A,c_alpha,G_alpha,h_alpha,X,Y,LandMarks,N)
    if i>0 and abs(v_A-v0)<Tol:
      break
    v0 = v_A
  return A,np.matrix(alpha)

### JSL Example #####
#X = [ np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
    #]
#Y = [ np.array([1, 1, 1, -1, -1, -1]) ]
#X = X+X
#Y = Y+[np.array([1,1,-1,1,-1,-1])]
#solvers.options['show_progress'] = False
#A,alpha = JSL(X,Y,X,lambda_val=1e-3,Diagonal=True)
#print X[0]*A*X[0].T*alpha[range(6),0]
#print X[0]*A*X[0].T*alpha[range(6,12),0]

#####################################################################
#### HingeLossLinear #####
# Method due to Guo & Ying (2014) Guaranteed Classification via Regularized Similarity Learning
# using L1 regularization and extended to multiple subproblems
# optimized using linear programming
  
def HingeLossLinear(X,Y,w=None,lambda_val=1.0,gamma=None,R=None,Diagonal=False,Symmetric=False):
  N = X[0].shape[1]
  N2 = N*N
  K = len(Y)
  totM = sum([len(Y_k) for Y_k in Y]) # total number of labeled examples
  if Diagonal: Symmetric=False
  if R==None: R = np.asmatrix(np.zeros((N,N)))
  if w==None: w = [1.0]*K
  if gamma==None: gamma = [1.0]*K
  w = [w[i]/float(len(Y[i])) for i in xrange(len(w))]
  landmarks = {}
  for k in xrange(K):
    for j in xrange(len(X[k])):
      if k in landmarks: landmarks[k] += X[k][j]*Y[k][j]
      else: landmarks[k] = X[k][j]*Y[k][j]
  row_grp = 0
  if not Diagonal: 
    c_A = [0]*N2+[0]*N2 + np.repeat(w,[len(Y_k) for Y_k in Y]).tolist()+[lambda_val]
    h_A = [-1]*totM+[0]*totM+ R.flatten().tolist()[0] + (-R).flatten().tolist()[0] + [0]*N + [0]
    G_A = spmatrix([],[],[],(2*totM+2*N2+N+1,2*N2+totM+1),tc='d')
    for k in xrange(K):
      X_k = X[k]
      Y_k = Y[k]
      L = landmarks[k]
      M = len(Y_k)
      D = float(M)*gamma[k]
      for i in xrange(M):
        G_A[row_grp+i,2*N2+row_grp+i] = -1
        G_A[totM+row_grp+i,2*N2+row_grp+i] = -1
        y_ki = Y_k[i]
        x = y_ki/D*X_k[i]
        for n in xrange(N):
          for m in xrange(N):
            G_A[row_grp+i,n*N+m] = -x[0,n]*L[0,m]
      row_grp += M
    for i in xrange(N2):
      G_A[2*totM+i,i] = 1
      G_A[2*totM+i,N2+i] = -1
      G_A[2*totM+N2+i,i] = -1
      G_A[2*totM+N2+i,N2+i] = -1
    for i in xrange(N):
      G_A[2*totM+2*N2+i,2*N2+totM] = -1
      for j in xrange(N):
        G_A[2*totM+2*N2+i,N2+i*N+j] = 1
    G_A[2*totM+2*N2+N,2*N2+totM] = -1
  else:
    c_A = [0]*N+[0]*N + np.repeat(w,[len(Y_k) for Y_k in Y]).tolist()+[lambda_val]
    h_A = [-1]*totM+[0]*totM+ np.diag(R).tolist() + np.diag(-R).tolist() + [0]*N + [0]
    G_A = spmatrix([],[],[],(2*totM+2*N+N+1,2*N+totM+1),tc='d')
    for k in xrange(K):
      X_k = X[k]
      Y_k = Y[k]
      L = landmarks[k]
      M = len(Y_k)
      D = float(M)*gamma[k]
      for i in xrange(M):
        G_A[row_grp+i,2*N+row_grp+i] = -1
        G_A[totM+row_grp+i,2*N+row_grp+i] = -1
        y_ki = Y_k[i]
        x = y_ki/D*X_k[i]
        for n in xrange(N):
          G_A[row_grp+i,n] = -x[0,n]*L[0,n]
      row_grp += M
    for i in xrange(N):
      G_A[2*totM+i,i] = 1
      G_A[2*totM+i,N+i] = -1
      G_A[2*totM+N+i,i] = -1
      G_A[2*totM+N+i,N+i] = -1
    for i in xrange(N):
      G_A[2*totM+2*N+i,2*N+totM] = -1
      G_A[2*totM+2*N+i,N+i] = 1
    G_A[2*totM+2*N+N,2*N+totM] = -1
  c_A = matrix(c_A,tc='d')
  h_A = matrix(h_A,tc='d')
  A_A = None
  b_A = None
  if not Diagonal and Symmetric:
    Nc2 = N*(N-1)/2
    A_A = spmatrix([],[],[],(Nc2,G_A.size[1]),tc='d')
    b_A = matrix(0,(Nc2,1),tc='d')
    r = 0
    for i in xrange(N):
      for j in xrange(i+1,N):
        A_A[r,i*N+j] = 1
        A_A[r,j*N+i] = -1
        r = r+1
  if not Symmetric: sol = solvers.lp(c_A,G_A,h_A)
  else: sol = solvers.lp(c_A,G_A,h_A,A_A,b_A)
  if sol['status']!='optimal':
    print 'WARNING! Optimization A step returned status: '+sol['status']
  if not Diagonal:
    return np.matrix(sol['x'][:N2]).reshape((N,N)) # expects x in row-major order
  else:
    return np.asmatrix(np.diag([sol['x'][i,0] for i in xrange(N)]))


### HingeLossLinear Example #####
#X = [ np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
    #]
#Y = [ np.array([1, 1, 1, -1, -1, -1]) ]
#X = X+X
#Y = Y+[np.array([1,1,-1,1,-1,-1])]
#A = HingeLossLinear(X,Y,lambda_val=1e-6,Symmetric=True)
#print X[0]*A*X[0].T
#print X[0]*A*(X[0].T*np.asmatrix(Y[0]).T)
#print X[1]*A*(X[1].T*np.asmatrix(Y[1]).T)

#####################################################################
#### RelativeHingeLoss #####
# Hinge loss over relative similarity constraints (triplet)
# using L1 regularization, extended to multiple subproblems
# due to Kyle Miller

def RelativeHingeLoss(X,w=None,lambda_val=1.0,R=None,Diagonal=False,Symmetric=False):
  # X is a list of subproblems, each subproblem is a list of triplets (a,b+,b-)
  # check training data
  N = X[0][0][0].shape[1]
  N2 = N*N
  K = len(X)
  for k in xrange(K):
    M = len(X[k])
    for i in xrange(M-1,0,-1):
      a = np.abs(X[k][i][0])
      b = np.abs(X[k][i][1]-X[k][i][2])
      if np.sum(a)==0:
        print "Warning: pivot in triplet "+str(i)+" of group "+str(k)+" is zero"
        print "This triplet will be ignored."
        del X[k][i]
      elif np.sum(b)==0:
        print "Warning: relative observations b+ and b- in triplet "+str(i)+" of group "+str(k)+" are equal"
        print "This triplet will be ignored."
        del X[k][i]
      elif Diagonal and np.sum(np.minimum(a,b))==0:
        print "Warning: a.*(b+ - b-) in triplet "+str(i)+" of group "+str(k)+" is the zero vector."
        print "This is problematic for diagonal problems."
        print "This triplet will be ignored."
        del X[k][i]
  totM = sum([len(X_k) for X_k in X]) # total number of labeled examples
  if Diagonal: Symmetric=False
  if R==None: R = np.asmatrix(np.zeros((N,N)))
  if w==None: w = [1.0]*K
  w = [w[i]/float(len(X[i])) for i in xrange(len(w))]
  row_grp = 0
  if not Diagonal: 
    c_A = [0]*N2+[0]*N2 + np.repeat(w,[len(X_k) for X_k in X]).tolist()+[lambda_val]
    h_A = [-1]*totM+[0]*totM+ R.flatten().tolist()[0] + (-R).flatten().tolist()[0] + [0]*N + [0]
    G_A = spmatrix([],[],[],(2*totM+2*N2+N+1,2*N2+totM+1),tc='d')
    for k in xrange(K):
      M = len(X[k])
      for i in xrange(M):
        G_A[row_grp+i,2*N2+row_grp+i] = -1
        G_A[totM+row_grp+i,2*N2+row_grp+i] = -1
        a = X[k][i][0]
        b = X[k][i][1]-X[k][i][2]
        for n in xrange(N):
          for m in xrange(N):
            G_A[row_grp+i,n*N+m] = -a[0,n]*b[0,m]
      row_grp += M
    for i in xrange(N2):
      G_A[2*totM+i,i] = 1
      G_A[2*totM+i,N2+i] = -1
      G_A[2*totM+N2+i,i] = -1
      G_A[2*totM+N2+i,N2+i] = -1
    for i in xrange(N):
      G_A[2*totM+2*N2+i,2*N2+totM] = -1
      for j in xrange(N):
        G_A[2*totM+2*N2+i,N2+i*N+j] = 1
    G_A[2*totM+2*N2+N,2*N2+totM] = -1
  else:
    c_A = [0]*N+[0]*N + np.repeat(w,[len(X_k) for X_k in X]).tolist()+[lambda_val]
    h_A = [-1]*totM+[0]*totM+ np.diag(R).tolist() + np.diag(-R).tolist() + [0]*N + [0]
    G_A = spmatrix([],[],[],(2*totM+2*N+N+1,2*N+totM+1),tc='d')
    for k in xrange(K):
      M = len(X[k])
      for i in xrange(M):
        G_A[row_grp+i,2*N+row_grp+i] = -1
        G_A[totM+row_grp+i,2*N+row_grp+i] = -1
        a = X[k][i][0]
        b = X[k][i][1]-X[k][i][2]
        for n in xrange(N):
          G_A[row_grp+i,n] = -a[0,n]*b[0,n]
      row_grp += M
    for i in xrange(N):
      G_A[2*totM+i,i] = 1
      G_A[2*totM+i,N+i] = -1
      G_A[2*totM+N+i,i] = -1
      G_A[2*totM+N+i,N+i] = -1
    for i in xrange(N):
      G_A[2*totM+2*N+i,2*N+totM] = -1
      G_A[2*totM+2*N+i,N+i] = 1
    G_A[2*totM+2*N+N,2*N+totM] = -1
  c_A = matrix(c_A,tc='d')
  h_A = matrix(h_A,tc='d')
  A_A = None
  b_A = None
  if not Diagonal and Symmetric:
    Nc2 = N*(N-1)/2
    A_A = spmatrix([],[],[],(Nc2,G_A.size[1]),tc='d')
    b_A = matrix(0,(Nc2,1),tc='d')
    r = 0
    for i in xrange(N):
      for j in xrange(i+1,N):
        A_A[r,i*N+j] = 1
        A_A[r,j*N+i] = -1
        r = r+1
  if not Symmetric: sol = solvers.lp(c_A,G_A,h_A)
  else: sol = solvers.lp(c_A,G_A,h_A,A_A,b_A)
  if sol['status']!='optimal':
    print 'WARNING! Optimization A step returned status: '+sol['status']
  if not Diagonal:
    return np.matrix(sol['x'][:N2]).reshape((N,N)) # expects x in row-major order
  else:
    return np.asmatrix(np.diag([sol['x'][i,0] for i in xrange(N)]))


### RelativeHingeLoss Example #####
#X = np.matrix([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
              #[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
              #[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
              #[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              #[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              #[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]])
#Y = [ np.array([1, 1, 1, -1, -1, -1]), np.array([1,1,-1,1,-1,-1]) ]
#T = [[],[]]
#for problem in xrange(2):
  #for i in xrange(len(X)):
    #for j in xrange(len(X)):
      #for k in xrange(len(X)):
        #if   Y[problem][i]==Y[problem][j] and Y[problem][i]!=Y[problem][k]:
          #T[problem].append( (X[i],X[j],X[k]) )
        #elif Y[problem][i]==Y[problem][k] and Y[problem][i]!=Y[problem][j]:
          #T[problem].append( (X[i],X[k],X[j]) )
        #if   Y[problem][j]==Y[problem][i] and Y[problem][j]!=Y[problem][k]:
          #T[problem].append( (X[j],X[i],X[k]) )
        #elif Y[problem][j]==Y[problem][k] and Y[problem][j]!=Y[problem][i]:
          #T[problem].append( (X[j],X[k],X[i]) )
        #if   Y[problem][k]==Y[problem][i] and Y[problem][k]!=Y[problem][j]:
          #T[problem].append( (X[k],X[i],X[j]) )
        #elif Y[problem][k]==Y[problem][j] and Y[problem][k]!=Y[problem][i]:
          #T[problem].append( (X[k],X[j],X[i]) )

#print len(T[0])
#A = RelativeHingeLoss(T,lambda_val=1e-3,Symmetric=False)
#res = X*A*X.T
## yes, yes
#print str(res[0,1])+" | "+str(res[0,2:]) # 0 and 1
#print str(res[1,0])+" | "+str(res[1,2:]) # 0 and 1

## yes, no or no, yes
#print str(res[2,3])+" | "+str(res[2,[0,1,4,5]]) # 2 and 3
#print str(res[3,2])+" | "+str(res[3,[0,1,4,5]]) # 3 and 2

## no, no
#print str(res[4,5])+" | "+str(res[4,[0,1,2,3]]) # 4 and 5
#print str(res[5,4])+" | "+str(res[5,[0,1,2,3]]) # 5 and 4

#print X*A*X[[0,2]].T
#print X*A*(X.T*np.asmatrix(Y[0]).T)
#print X*A*(X.T*np.asmatrix(Y[1]).T)
