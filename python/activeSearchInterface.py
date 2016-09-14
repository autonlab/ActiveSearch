from __future__ import division, print_function
import time
import numpy as np, numpy.linalg as nlg, numpy.random as nr
import scipy.sparse as ss, scipy.linalg as slg, scipy.sparse.linalg as ssl

import IPython

np.set_printoptions(suppress=True, precision=5, linewidth=100)

def matrix_squeeze(X):
  # converts into numpy.array and squeezes out singular dimensions
  return np.squeeze(np.asarray(X))

class Parameters(object):

  def __init__ (self, pi=0.05, eta=0.5, w0=None, alpha=0, sparse=True, verbose=True, remove_self_degree=False):
    """
    pi      --> prior target probability
    eta     --> jump probability
    w0      --> weight assigned to regularization on unlabeled points (default: 1/(#data points))
    """
    self.pi = pi
    self.eta = eta
    self.w0 = w0
    self.sparse = sparse
    self.remove_self_degree = remove_self_degree

    self.verbose = verbose

    self.alpha = alpha

## For more on how these functions operate, see their analogs in daemon_service.py
class genericAS(object):

  def __init__ (self, params=Parameters()):
    """
    pi      --> prior target probability
    eta     --> jump probability
    w0      --> 
    """
    self.params = params

    self.reset()

  def reset(self):
    self.start_point = None
    self.unlabeled_idxs = []
    self.labeled_idxs = []
    self.next_message = None

    self.seen_next = False

    # Number of times the algorithm sets something to be +ve
    # Maybe it needs to be incremented only when it sets something it requested
    self.hits = [] # not entirely sure what happens to this if allowed to reset
    self.labels = []

    self.iter = -1
    self.initialized = False

  def saveState (self, filename):
    # save the state of the parameters and the data so that we can restart.
    raise NotImplementedError()

  def firstMessage(self, idx):
    # Assuming this is always +ve. Can be changed otherwise
    # Need to check whether this does the right thing.
    if not self.initialized:
      raise Exception ("Has not been initialized with data")

    if self.iter >= 0:
      print("First message has already been set. Treating this as a positive.")
    else:
      self.start_point = idx
    self.setLabel(idx, 1)

  def interestingMessage(self):
    if self.next_message is None:
      if self.iter < 0:
        raise Exception("The algortithm has not been initialized. There is no current message.")
      else:
        raise Exception("I don't know how you got here.")
    if not self.seen_next:
      raise Exception ("This message has not been requested/seen yet.")
    self.setLabel(self.next_message, 1)

  def boringMessage(self):
    if self.next_message is None:
      if self.iter < 0:
        raise Exception("The algortithm has not been initialized. There is no current message.")
      else:
        raise Exception("I don't know how you got here.")
    if not self.seen_next:
      raise Exception ("This message has not been requested/seen yet.")
    self.setLabel(self.next_message, 0)

  def setalpha(self, alpha):
    self.params.alpha = alpha

  def getStartPoint (self):
    if self.start_point is None:
      raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
    return self.start_point

  def setLabel (self, idx, lbl):
    raise NotImplementedError()

  def resetLabel (self, idx, lbl):
    raise NotImplementedError()

  def setLabelCurrent(self, value):
    if self.next_message is None:
      if self.iter < 0:
        raise Exception("The algortithm has not been initialized. There is no current message.")
      else:
        raise Exception("I don't know how you got here.")
    if not self.seen_next:
      raise Exception ("This message has not been requested/seen yet.")
    self.setLabel(self.next_message, value)

  def getNextMessage (self):
    if self.next_message is None:
      if self.iter < 0:
        raise Exception("The algortithm has not been initialized. There is no current message.")
      else:
        raise Exception("I don't know how you got here.")
    self.seen_next = True
    return self.next_message

  def setLabelBulk (self, idxs, lbls):
    for idx,lbl in zip(idxs,lbls):
      self.setLabel(idx,lbl)

  def pickRandomLabeledMessage (self):
    if self.iter < 0:
      raise Exception("The algortithm has not been initialized. Please call \"firstMessage\".")
    return self.labeled_idxs[nr.randint(len(self.labeled_idxs))]

  def getLabel (self, idx):
    # 0 is boring, 1 is interesting and -1 is unlabeled
    return self.labels[idx]


class linearizedAS (genericAS):

  def __init__ (self, params=Parameters()):
    genericAS.__init__ (self, params)

  def initialize(self, Xf, init_labels = {}):
    """
    Xf      --> r x n matrix of feature values for each point.
            where r is # features, n is # points.
    init_labels --> dictionary from index to label of initial labels.
    """
    self.reset()

    # Save Xf and initialize some of the variables which depend on Xf
    self.Xf = Xf
    self.r, self.n = Xf.shape

    self.labeled_idxs = init_labels.keys()
    self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

    self.labels = np.array([-1]*self.n)
    for lidx in self.labeled_idxs:
      self.labels[lidx] = init_labels[lidx]

    # Initialize some parameters and constants which are needed and not yet initialized
    if self.params.sparse:
      self.Ir = ss.eye(self.r)
    else:
      self.Ir = np.eye(self.r)

    self.l = (1-self.params.eta)/self.params.eta
    if self.params.w0 is None:
      self.params.w0 = 1/self.n

    # Set up some of the initial values of some matrices needed to compute D, BDinv, q and f
    B = np.where(self.labels==-1, 1/(1+self.params.w0),self.l/(1+self.l))
    # B[self.labeled_idxs] = self.l/(1+self.l)
    D = np.squeeze(Xf.T.dot(Xf.dot(np.ones((self.n,1)))))
    if self.params.remove_self_degree:
      Ds = matrix_squeeze((Xf.multiply(Xf)).sum(0))
      D = D - Ds

    self.Dinv = 1./D

    self.BDinv = np.squeeze(B*self.Dinv)
    if self.params.sparse:
      self.BDinv_ss = ss.diags([np.squeeze(B*self.Dinv)],[0]).tocsr()
    
    self.q = (1-B)*np.where(self.labels==-1,self.params.pi,self.labels) # Need to update q every iteration
    #self.q[self.labeled_idxs] *= np.array(init_labels.values())/self.params.pi

    # Constructing and inverting C
    if self.params.verbose:
      print("Constructing C")
      t1 = time.time()
    if self.params.sparse:
      C = (self.Ir - self.Xf.dot(self.BDinv_ss.dot(self.Xf.T))) 
    else:
      C = (self.Ir - self.Xf.dot(self.BDinv[:,None]*self.Xf.T))
    if self.params.verbose:
      print("Time for constructing C:", time.time() - t1)

    if self.params.verbose:
      print ("Inverting C")
      t1 = time.time()
    # Our matrix is around 40% sparse which makes ssl.inv run very slowly. We will just use the regular nlg.inv
    if self.params.sparse:
      self.Cinv = ss.csr_matrix(nlg.inv(C.todense())) # Need to update Cinv every iteration
    else:
      self.Cinv = nlg.inv(C)

    if self.params.verbose: 
      print("Time for inverse:", time.time() - t1)

    # Just keeping this around. Don't really need it.
    if self.params.sparse:
      self.f = self.q + self.BDinv_ss.dot(((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.q))))))
    else:
      self.f = self.q + self.BDinv*((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.q)))))

    # Impact factor calculations
    if self.params.alpha > 0:
      # 0. Some useful variables
      self.dP = (1./self.l-self.params.w0)*D # L - U
      self.dPpi = (1./self.l-self.params.pi*self.params.w0)*D # L - pi*U
    
      # 1. Df_tilde
      # First, we need J = diag (X^T * Cinv * X): each element of J is x_i^T*Cinv*x_i
      if self.params.sparse:
        self.J = matrix_squeeze(((self.Cinv.dot(self.Xf)).multiply(self.Xf)).sum(0))
      else:
        self.J = np.squeeze(((self.Cinv.dot(self.Xf))*self.Xf).sum(0))
      # Now we compute the entire diag
      diagMi = (1+self.BDinv*self.J)*self.BDinv
      # Finally, Df_tilde
      dpf = (self.dPpi - self.dP*self.f)
      Df_tilde = dpf*diagMi/(1 + self.dP*diagMi)

      # 2. DF
      # z = (I-B)Pinv*u = B*Dinv*u (these are defined in Kernel AS notes)
      self.z = np.where(self.labels==-1, self.BDinv, 0)
      if self.params.sparse:
        Minv_u = self.z + self.BDinv_ss.dot(self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.z))))
      else:
        Minv_u = self.z + self.BDinv*(self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.z))))
      
      DF = (dpf - self.dP*Df_tilde)*Minv_u
      
      # 3. IM
      self.IM = self.f*(DF-Df_tilde)
      
      # Normalize IM
      self.IM = self.IM * self.f.mean() / self.IM.mean()

    # Setting iter/start_point
    # If batch initialization is done, then start_point is everything given
    if len(self.labeled_idxs) > 0:
      if len(self.labeled_idxs) == 0:
        self.start_point = self.labeled_idxs[0]
      else:
        self.start_point = [eid for eid in self.labeled_idxs]
      # Finding the next message to show -- get the current max element
      if self.params.alpha > 0:
        uidx = np.argmax((self.f+self.params.alpha*self.IM)[self.unlabeled_idxs])
      else:
        uidx = np.argmax(self.f[self.unlabeled_idxs])
      self.next_message = self.unlabeled_idxs[uidx]
      # Now that a new message has been selected, mark it as unseen
      self.seen_next = False 

      self.iter = 0
      self.hits = [sum(init_labels.values())]

    if self.params.verbose:
      print("Done with the initialization.")

    self.initialized = True

  def setLabel (self, idx, lbl, display_iter = None):
    # Set label for given message id

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = 0 if lbl <= 0 else 1
  
    # First, some book-keeping
    # If setLabel is called without "firstMessage," then set start_point
    if self.start_point is None:
      self.start_point = idx
    self.iter += 1
    self.labels[idx] = lbl
    self.unlabeled_idxs.remove(idx)

    # Updating various parameters to calculate next C inverse and f
    self.BDinv[idx] = self.l/(1+self.l) * self.Dinv[idx]
    if self.params.sparse:
      self.BDinv_ss[idx,idx] = self.l/(1+self.l) * self.Dinv[idx]

    self.q[idx] = lbl*1/(1+self.l)
    gamma = -(self.l/(1+self.l)-1/(1+self.params.w0))*self.Dinv[idx]

    Xi = self.Xf[:,[idx]] # ith feature vector
    Cif = self.Cinv.dot(Xi)

    if self.params.sparse:
      c =  np.squeeze(gamma/(1 + (gamma*Xi.T.dot(Cif))[0,0]))
    else:
      c =  np.squeeze(gamma/(1 + gamma*Xi.T.dot(Cif)))
    self.Cinv = self.Cinv - c*(Cif.dot(Cif.T))
  
    if self.params.sparse:
      self.f = self.q + self.BDinv_ss.dot(((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.q))))))
    else:
      self.f = self.q + self.BDinv*((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.q)))))

    # Updating IM
    if self.params.alpha > 0:
      self.z[idx] = 0
      if self.params.sparse:
        Minv_u = self.z + self.BDinv_ss.dot(self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.z))))
      else:
        Minv_u = self.z + self.BDinv*(self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.z))))
      dpf = (self.dPpi - self.dP*self.f)
      # Updating Df_tilde
      if self.params.sparse:
        self.J = self.J - c*(matrix_squeeze((self.Xf.T.dot(Cif)).todense())**2)
      else:
        self.J = self.J - c*(np.squeeze(self.Xf.T.dot(Cif))**2)
      diagMi = (1+self.BDinv*self.J)*self.BDinv
      Df_tilde = dpf*diagMi/(1 + self.dP*diagMi)
      # Updating DF
      DF = (dpf - self.dP*Df_tilde)*Minv_u
      # Computing IM
      self.IM = self.f*(DF-Df_tilde)
      
      # Normalize IM
      self.IM = self.IM * self.f.mean() / self.IM.mean()

    # Some more book-keeping
    # Not sure how we're going to change "history" in some sense:
    # -- everything since this was first incorrectly labeled.
    self.labeled_idxs.append(idx)
    if self.iter == 0:
      self.hits.append(lbl)
    else:
      self.hits.append(self.hits[-1] + lbl)

    # Finding the next message to show -- get the current max element
    if self.params.alpha > 0:
      uidx = np.argmax((self.f + self.params.alpha*self.IM)[self.unlabeled_idxs])
    else:
      uidx = np.argmax(self.f[self.unlabeled_idxs])
    self.next_message = self.unlabeled_idxs[uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      elapsed = time.time() - t1
      display_iter = display_iter if display_iter else self.iter
      print( 'Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (display_iter, self.labeled_idxs[-1], self.hits[-1], elapsed))

  # Need to think a bit about the math here
  # we return the old label
  def resetLabel (self, idx, lbl):
    # Reset label for given message id

    # If reset label is called on something not yet set, it should do the same as setLabel
    ret = self.labels[idx]
    if self.labels[idx] == -1:
      self.setLabel(idx, lbl)
      return ret 
    elif self.labels[idx] == lbl:
      print("Already the same value!")
      return ret

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = 0 if lbl <= 0 else 1
  
    # First, some book-keeping
    # Message is already labeled, se we don't change self.unlabeled_idx
    self.iter += 1 # We're increasing iter because it will be consistent with "hits"
    self.labels[idx] = lbl

    # Updating various parameters to calculate next f
    # Cinv is already correct as it does not depend on the label -- just whether the nodes are labeled or not
    # self.q[idx] = lbl*self.l/(1+self.l) --> old
    gamma = (lbl-self.labels[idx])*self.l/(1+self.l)
    ei = np.zeros((self.n,1))
    ei[idx] = 1
  
    # f = q + Aq + gamma*(ei + Aei)
    if self.params.sparse:
      self.f = self.f + gamma*self.BDinv_ss.dot(((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(ei))))))
    else:
      self.f = self.f + gamma*self.BDinv*((self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(ei)))))
    self.f[idx] += gamma
    self.q[idx] += gamma

    # Updating IM
    if self.params.alpha > 0:
      Xi = self.Xf[:,[idx]] # ith feature vector
      Cif = self.Cinv.dot(Xi)

      gamma2 = -(self.l/(1+self.l)-1/(1+self.params.w0))*self.Dinv[idx]
      if self.params.sparse:
        c =  np.squeeze(gamma2/(1 + (gamma2*Xi.T.dot(Cif))[0,0]))
      else:
        c =  np.squeeze(gamma2/(1 + gamma2*Xi.T.dot(Cif)))

      self.z[idx] = 0
      if self.params.sparse:
        Minv_u = self.z + self.BDinv_ss.dot(self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.z))))
      else:
        Minv_u = self.z + self.BDinv*(self.Xf.T.dot(self.Cinv.dot(self.Xf.dot(self.z))))
      dpf = (self.dPpi - self.dP*self.f)
      # Updating Df_tilde
      if self.params.sparse:
        self.J = self.J - c*(matrix_squeeze((self.Xf.T.dot(Cif)).todense())**2)
      else:
        self.J = self.J - c*(np.squeeze(self.Xf.T.dot(Cif))**2)
      diagMi = (1+self.BDinv*self.J)*self.BDinv
      Df_tilde = dpf*diagMi/(1 + self.dP*diagMi)
      # Updating DF
      DF = (dpf - self.dP*Df_tilde)*Minv_u
      # Computing IM
      self.IM = self.f*(DF-Df_tilde)

    # Some more book-keeping
    self.hits.append(self.hits[-1] + (-1 if lbl == 0 else 1))

    # Finding the next message to show -- get the current max element
    if self.params.alpha > 0:
      uidx = np.argmax((self.f+self.params.alpha*self.IM)[self.unlabeled_idxs])
    else:
      uidx = np.argmax(self.f[self.unlabeled_idxs])
    self.next_message = self.unlabeled_idxs[uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 
    # this is confusing. What if they reset something before they mark "current email" as something? Not sure

    if self.params.verbose:
      elapsed = time.time() - t1
      print('Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (self.iter, self.labeled_idxs[-1], self.hits[-1], elapsed))

    return ret

########## Weighted Neighbor Active Search

class WNParameters (Parameters):

  def __init__ (self, normalize=True, use_prior=False, pi=0.5, prior_weight=0.1, sparse=True, verbose=True):
    """
    Parameters specifically for nearest neighbours.
    """
    Parameters.__init__ (self, sparse=sparse, verbose=verbose, pi=pi)
    self.normalize = normalize
    self.use_prior = use_prior
    self.prior_weight = prior_weight
    if self.pi is not None:
      self.pi = (self.pi * 2) - 1


class weightedNeighborAS (genericAS):

  def __init__ (self, params=WNParameters()):
    genericAS.__init__(self, params)

  def initialize(self, Xf, init_labels = {}):
    """
    Xf      --> r x n matrix of feature values for each point.
            where r is # features, n is # points.
    init_labels --> dictionary from index to label of initial labels.
    """
    # Save Xf and initialize some of the variables which depend on Xf
    self.Xf = Xf
    self.r, self.n = Xf.shape

    self.labeled_idxs = init_labels.keys()
    self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

    self.labels = np.array([0]*self.n)
    for lidx in self.labeled_idxs:
      self.labels[lidx] = 1 if init_labels[lidx] == 1 else -1

    self.NN_avg_similarity  = None
    if self.params.normalize or self.params.use_prior:
      self.NN_abs_similarity = None
    if self.params.use_prior:
      self_similarity = (self.Xf[:, self.unlabeled_idxs] *
                         self.Xf[:, self.unlabeled_idxs]).sum(1).squeeze()
      self.NN_prior_similarity = self.Xf[:, self.unlabeled_idxs].T.dot(
          self.Xf.dot(np.ones(self.n))) - self_similarity

    # Setting iter/start_point
    # If batch initialization is done, then start_point is everything given
    if len(self.labeled_idxs) > 0:
      self.start_point = [eid for eid in self.labeled_idxs]
      
      ## Computing KNN similarity
      NN_similarity = self.Xf[:, self.unlabeled_idxs].T.dot(self.Xf[:, self.labeled_idxs])
      self.NN_avg_similarity = NN_similarity.dot(self.labels[self.labeled_idxs]).squeeze()
      if self.params.use_prior:
        self.NN_abs_similarity = np.array(np.abs(NN_similarity).sum(1)).squeeze()
        self.NN_prior_similarity -= self.NN_abs_similarity
        self.f = self.NN_avg_similarity + self.NN_prior_similarity * self.params.pi
        if self.params.normalize:  
          self.f /= (self.params.prior_weight * self.NN_prior_similarity + self.NN_abs_similarity)
          self.f = np.nan_to_num(self.f)
      else:
        self.f = self.NN_avg_similarity.copy()
        if self.params.normalize:
          self.NN_abs_similarity = np.array(np.abs(NN_similarity).sum(1)).squeeze()
          self.f /= self.NN_abs_similarity
          self.f = np.nan_to_num(self.f)


      # Finding the next message to show -- get the current max element
      self.uidx = np.argmax(self.f)
      self.next_message = self.unlabeled_idxs[self.uidx]
      # Now that a new message has been selected, mark it as unseen
      self.seen_next = False 

      self.iter = 0
      self.hits = [sum([lbl == 1 for lbl in init_labels.values()])]

    if self.params.verbose:
      print("Done with the initialization.")

    self.initialized = True

  def setLabel (self, idx, lbl, display_iter = None):
    # Set label for given message id
    # Always assuming idx has not been labeled yet.
    ## BUT HERE IF SOMETHING HAS NOT BEEN LABELED YET

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = -1 if lbl <= 0 else 1
  
    # First, some book-keeping
    # If setLabel is called without "firstMessage," then set start_point
    if self.start_point is None:
      self.start_point = idx
    if self.next_message != idx:
      self.uidx = np.nonzero(np.array(self.unlabeled_idxs)==idx)[0][0]

    self.iter += 1
    self.labels[idx] = lbl
    self.unlabeled_idxs.remove(idx)

    # Updating various parameters to calculate f
    Xi = self.Xf[:,[idx]] # ith feature vector
    Sif = self.Xf[:, self.unlabeled_idxs].T.dot(Xi)
    if self.params.sparse:
      Sif = matrix_squeeze(Sif.todense())
    else:
      Sif = Sif.squeeze()

    if self.NN_avg_similarity is None:
      self.NN_avg_similarity = Sif * lbl
      self.f = self.NN_avg_similarity.copy()
      # TODO: add code for use_prior
      if self.params.normalize:
        self.NN_abs_similarity = Sif
        self.f /= self.NN_abs_similarity
        self.f = np.nan_to_num(self.f)
    else:
      self.NN_avg_similarity = np.delete(self.NN_avg_similarity, self.uidx) + lbl*Sif
      if self.params.use_prior:
        self.NN_abs_similarity = np.delete(self.NN_abs_similarity, self.uidx) + np.abs(Sif)
        self.NN_prior_similarity = np.delete(self.NN_prior_similarity, self.uidx) - self.NN_abs_similarity
        self.f = self.NN_avg_similarity + self.NN_prior_similarity * self.params.pi
        if self.params.normalize:
          self.f /= (self.params.prior_weight * self.NN_prior_similarity + self.NN_abs_similarity)
          self.f = np.nan_to_num(self.f)
      else:
        self.f = self.NN_avg_similarity.copy()
        if self.params.normalize:
          self.NN_abs_similarity = np.delete(self.NN_abs_similarity, self.uidx) + np.abs(Sif)
          self.f /= self.NN_abs_similarity
          self.f = np.nan_to_num(self.f)

    # Some more book-keeping
    self.labeled_idxs.append(idx)
    if self.iter == 0:
      self.hits.append(max(0, lbl))
    else:
      self.hits.append(self.hits[-1] + max(0, lbl))

    # Finding the next message to show -- get the current max element
    # import IPython
    # IPython.embed()
    self.uidx = np.argmax(self.f)
    self.next_message = self.unlabeled_idxs[self.uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      self.elapsed = time.time() - t1
      display_iter = display_iter if display_iter else self.iter
      print( 'NNAS -- Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (display_iter, self.labeled_idxs[-1], self.hits[-1], self.elapsed))

  def resetLabel (self, idx, lbl):
    raise NotImplementedError('resetLabel not implemented')
    # TODO: Change this for incorporating prior
    # Reset label for given message id
    # If reset label is called on something not yet set, it should do the same as setLabel
    ret = self.labels[idx]
    if self.labels[idx] == -1:
      self.setLabel(idx, lbl)
      return ret 
    elif self.labels[idx] == lbl:
      print("Already the same value!")
      return ret

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = -1 if lbl <= 0 else 1
  
    # Updating various parameters to calculate and f
    Xi = self.Xf[:,[idx]] # ith feature vector
    Sif = self.Xif[:, self.unlabeled_idxs].T.dot(Xi)

    if self.params.normalize:
      self.NN_avg_similarity += 2*lbl*Sif
      self.f = self.NN_avg_similarity*(1/self.NN_abs_similarity)
      self.f = np.nan_to_num(self.f)
    else:
      self.f += 2*lbl*Sif

    # First, some book-keeping
    # Message is already labeled, se we don't change self.unlabeled_idx
    self.iter += 1 # We're increasing iter because it will be consistent with "hits"
    self.labels[idx] = lbl

    # Some more book-keeping
    # Not sure how we're going to change "history" in some sense:
    # -- everything since this was first incorrectly labeled.
    self.hits.append(self.hits[-1] + (-1 if lbl == 0 else 1))

    # Finding the next message to show -- get the current max element
    uidx = np.argmax(self.f)
    self.next_message = self.unlabeled_idxs[self.uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      self.elapsed = time.time() - t1
      print('Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (self.iter, self.labeled_idxs[-1], self.hits[-1], self.elapsed))

    return ret


class weightedNeighborGraphAS (genericAS):

  def __init__ (self, params=WNParameters()):
    genericAS.__init__ (self, params)

  def initialize(self, A, init_labels = {}):
    """
    A       --> affinity matrix of graph
    init_labels --> dictionary from index to label of initial labels.
    """
    # Save A and initialize some of the variables which depend on A
    self.A = A
    self.n = A.shape[0]

    self.labeled_idxs = init_labels.keys()
    self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

    self.labels = np.array([0]*self.n)
    for lidx in self.labeled_idxs:
      self.labels[lidx] = 1 if init_labels[lidx] == 1 else -1

    
    self.NN_avg_similarity  = None
    if self.params.normalize or self.params.use_prior:
      self.NN_abs_similarity = None
    if self.params.use_prior:
      self_similarity = np.diag(A)[self.unlabeled_idxs]
      self.NN_prior_similarity = self.A[self.unlabeled_idxs,:].sum(1) - self_similarity

    # Setting iter/start_point
    # If batch initialization is done, then start_point is everything given
    if len(self.labeled_idxs) > 0:
      self.start_point = [eid for eid in self.labeled_idxs]

      ## Computing KNN similarity
      NN_similarity = self.A[np.ix_(self.unlabeled_idxs, self.labeled_idxs)]
      self.NN_avg_similarity = NN_similarity.dot(self.labels[self.labeled_idxs]).squeeze()
      if self.params.use_prior:
        self.NN_abs_similarity = np.array(np.abs(NN_similarity).sum(1)).squeeze()
        self.NN_prior_similarity -= self.NN_abs_similarity
        self.f = self.NN_avg_similarity + self.NN_prior_similarity * self.params.pi
        if self.params.normalize:  
          self.f /= (self.params.prior_weight * self.NN_prior_similarity + self.NN_abs_similarity)
          self.f = np.nan_to_num(self.f)
      else:
        self.f = self.NN_avg_similarity.copy()
        if self.params.normalize:
          self.NN_abs_similarity = np.array(np.abs(NN_similarity).sum(1)).squeeze()
          self.f /= self.NN_abs_similarity
          self.f = np.nan_to_num(self.f)

      # Finding the next message to show -- get the current max element
      self.uidx = np.argmax(self.f)
      self.next_message = self.unlabeled_idxs[self.uidx]
      # Now that a new message has been selected, mark it as unseen
      self.seen_next = False 

      self.iter = 0
      self.hits = [sum([lbl == 1 for lbl in init_labels.values()])]

    if self.params.verbose:
      print("Done with the initialization.")

    self.initialized = True

  def setLabel (self, idx, lbl, display_iter = None):
    # Set label for given message id
    # Always assuming idx has not been labeled yet.
    ## BUT HERE IF SOMETHING HAS NOT BEEN LABELED YET

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = -1 if lbl <= 0 else 1
  
    # First, some book-keeping
    # If setLabel is called without "firstMessage," then set start_point
    if self.start_point is None:
      self.start_point = idx
    if self.next_message != idx:
      self.uidx = np.nonzero(np.array(self.unlabeled_idxs)==idx)[0][0]

    self.iter += 1
    self.labels[idx] = lbl
    self.unlabeled_idxs.remove(idx)

    # Updating various parameters to calculate f
    Sif = self.A[self.unlabeled_idxs, idx]
    if self.params.sparse:
      Sif = matrix_squeeze(Sif.todense())

    self.NN_avg_similarity = np.delete(self.NN_avg_similarity, self.uidx) + lbl*Sif
    if self.params.use_prior:
      self.NN_abs_similarity = np.delete(self.NN_abs_similarity, self.uidx) + np.abs(Sif)
      self.NN_prior_similarity = np.delete(self.NN_prior_similarity, self.uidx) - self.NN_abs_similarity
      self.f = self.NN_avg_similarity + self.NN_prior_similarity * self.params.pi
      if self.params.normalize:
        self.f /= (self.params.prior_weight * self.NN_prior_similarity + self.NN_abs_similarity)
        self.f = np.nan_to_num(self.f)
    else:
      self.f = self.NN_avg_similarity.copy()
      if self.params.normalize:
        self.NN_abs_similarity = np.delete(self.NN_abs_similarity, self.uidx) + np.abs(Sif)
        self.f /= self.NN_abs_similarity
        self.f = np.nan_to_num(self.f)

    # Some more book-keeping
    self.labeled_idxs.append(idx)
    if self.iter == 0:
      self.hits.append(max(0, lbl))
    else:
      self.hits.append(self.hits[-1] + max(0, lbl))

    # Finding the next message to show -- get the current max element
    self.uidx = np.argmax(self.f)
    self.next_message = self.unlabeled_idxs[self.uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      self.elapsed = time.time() - t1
      display_iter = display_iter if display_iter else self.iter
      print( 'NNAS -- Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (display_iter, self.labeled_idxs[-1], self.hits[-1], self.elapsed))

  def resetLabel (self, idx, lbl):
    raise NotImplementedError('resetLabel not implemented')
    # TODO: Change this for incorporating prior
    # Reset label for given message id
    # If reset label is called on something not yet set, it should do the same as setLabel
    ret = self.labels[idx]
    if self.labels[idx] == -1:
      self.setLabel(idx, lbl)
      return ret 
    elif self.labels[idx] == lbl:
      print("Already the same value!")
      return ret

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = -1 if lbl <= 0 else 1
  
    # Updating various parameters to calculate and f
    Sif = self.A[self.unlabeled_idxs, idx]
    if self.params.sparse:
      Sif = matrix_squeeze(Sif.todense())

    if self.params.normalize:
      self.NN_avg_similarity += 2*lbl*Sif
      self.f = self.NN_avg_similarity*(1/self.self.NN_abs_similarity)
      self.f = np.nan_to_num(self.f)
    else:
      self.f += 2*lbl*Sif

    # First, some book-keeping
    # Message is already labeled, se we don't change self.unlabeled_idx
    self.iter += 1 # We're increasing iter because it will be consistent with "hits"
    self.labels[idx] = lbl

    # Some more book-keeping
    # Not sure how we're going to change "history" in some sense:
    # -- everything since this was first incorrectly labeled.
    self.hits.append(self.hits[-1] + (-1 if lbl == 0 else 1))

    # Finding the next message to show -- get the current max element
    uidx = np.argmax(self.f)
    self.next_message = self.unlabeled_idxs[self.uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      self.elapsed = time.time() - t1
      print('Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (self.iter, self.labeled_idxs[-1], self.hits[-1], self.elapsed))

    return ret


class shariAS (genericAS):

  def __init__ (self, params=Parameters()):
    genericAS.__init__ (self, params)

  def initialize(self, A, init_labels = {}):
    """
    A       --> n x n affinity matrix of feature values for each point.
    init_labels --> dictionary from index to label of initial labels.
    """
    # Save Xf and initialize some of the variables which depend on Xf
    self.A = A
    self.n = A.shape[0]

    self.labeled_idxs = init_labels.keys()
    self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

    self.labels = np.array([-1]*self.n)
    self.labels[self.labeled_idxs] = init_labels.values()

    # Initialize some parameters and constants which are needed and not yet initialized
    self.l = (1-self.params.eta)/self.params.eta
    if self.params.w0 is None:
      self.params.w0 = 1/self.n

    # Set up some of the initial values of some matrices
    B = np.where(self.labels==-1, 1/(1+self.params.w0),self.l/(1+self.l))
    #B = np.ones(self.n)/(1 + self.params.w0) ##
    D = matrix_squeeze(self.A.sum(1)) ##
    if self.params.remove_self_degree:
      if self.params.verbose:
        print("Removing diagonal elements.")
      D -= A.diagonal()
      self.A = self.A - np.diag(matrix_squeeze(A.diagonal()))
    self.Dinv = 1./D

    if self.params.sparse:
      BDinv = ss.diags([np.squeeze(B*self.Dinv)],[0]).tocsr()
      I_A = ss.diags([np.ones(self.n)],[0]).tocsr()-BDinv.dot(self.A)
    else:
      I_A = -np.squeeze(B*self.Dinv)[:,None]*self.A
      I_A[xrange(self.n), xrange(self.n)] += 1 ##

    # Constructing and inverting I - A'
    if self.params.verbose:
      print ("Inverting I_A")
      t1 = time.time()

    if self.params.sparse:
      self.I_A_inv = ss.csr_matrix(nlg.inv(I_A.todense()))
    else:
      self.I_A_inv = np.matrix(nlg.inv(I_A))

    if self.params.verbose:
      print("Time for inverse:", time.time() - t1)

    # Need to update q every iteration
    q = (1-B)*np.where(self.labels==-1,self.params.pi,self.labels)
    #q = (1-B)*self.params.pi*np.ones(self.n)
    self.f = matrix_squeeze(self.I_A_inv.dot(q))
    
    # Setting iter/start_point
    # If batch initialization is done, then start_point is everything given
    if len(self.labeled_idxs) > 0:
      if len(self.labeled_idxs) == 0:
        self.start_point = self.labeled_idxs[0]
      else:
        self.start_point = [eid for eid in self.labeled_idxs]

      # Finding the next message to show -- get the current max element
      uidx = np.argmax(self.f[self.unlabeled_idxs])
      self.next_message = self.unlabeled_idxs[uidx]
      # Now that a new message has been selected, mark it as unseen
      self.seen_next = False 

      self.iter = 0
      self.hits = [sum(init_labels.values())]
    
    if self.params.verbose:
      print ("Done with the initialization.")
    
    self.initialized = True

  def setLabel (self, idx, lbl):
    # Set label for given message id

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = 0 if lbl <= 0 else 1
  
    # First, some book-keeping
    # If setLabel is called without "firstMessage," then set start_point
    if self.start_point is None:
      self.start_point = idx
    self.iter += 1
    self.labels[idx] = lbl
    self.unlabeled_idxs.remove(idx)

    # Keeping some constants around
    p1 = self.params.w0*self.params.pi/(1+ self.params.w0)

    t = (1+self.params.w0)*(1-self.params.eta)
    s = self.params.eta*lbl - p1

    # More constants
    BDinv_Ai = self.Dinv[idx]/(1+ self.params.w0)*self.A[idx,:]

    if self.params.sparse:
      p2 = (1+(1-t)*matrix_squeeze(BDinv_Ai.dot(self.I_A_inv[:,idx]).todense()[0]))
    else:
      p2 = (1+(1-t)*matrix_squeeze(BDinv_Ai.dot(self.I_A_inv[:,idx])))

    fdel = (s - (1-t)*(self.f[idx]-p1))/p2
    IAdel =  - (1-t)*self.I_A_inv[:,idx].dot(BDinv_Ai.dot(self.I_A_inv))/p2

    if self.params.sparse:
      self.f += fdel*matrix_squeeze(self.I_A_inv[:,idx].todense())
    else:
      self.f += fdel*matrix_squeeze(self.I_A_inv[:,idx])
    self.I_A_inv = self.I_A_inv + IAdel

    # Some more book-keeping
    self.labeled_idxs.append(idx)
    if self.iter == 0:
      self.hits.append(lbl)
    else:
      self.hits.append(self.hits[-1] + lbl)

    # Finding the next message to show -- get the current max element
    uidx = np.argmax(self.f[self.unlabeled_idxs])
    self.next_message = self.unlabeled_idxs[uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      elapsed = time.time() - t1
      print('Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (self.iter, self.labeled_idxs[-1], self.hits[-1], elapsed))


class naiveAS (genericAS):

  def __init__ (self, params=Parameters()):
    genericAS.__init__ (self, params)

  def initialize(self, A, init_labels = {}):
    """
    A       --> n x n affinity matrix of feature values for each point.
    """
    # Save Xf and initialize some of the variables which depend on Xf
    self.A = A
    self.n = A.shape[0]

    self.labeled_idxs = init_labels.keys()
    self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

    self.labels = np.array([-1]*self.n)
    self.labels[self.labeled_idxs] = init_labels.values()

    # Initialize some parameters and constants which are needed and not yet initialized
    self.l = (1-self.params.eta)/self.params.eta
    if self.params.w0 is None:
      self.params.w0 = 1/self.n

    # Set up some of the initial values of some matrices
    B = np.where(self.labels==-1, 1/(1+self.params.w0),self.l/(1+self.l))
    D = np.squeeze(self.A.sum(1)) ##
    self.Dinv = 1./D
    self.BDinv = np.diag(np.squeeze(B*self.Dinv))

    # Need to update q every iteration
    self.q = (1-B)*np.where(self.labels==-1,self.params.pi,self.labels)
    I_A = np.eye(self.n) - self.BDinv.dot(self.A)

    self.f =  nlg.solve(I_A, self.q)
    # Setting iter/start_point
    # If batch initialization is done, then start_point is everything given
    if len(self.labeled_idxs) > 0:
      if len(self.labeled_idxs) == 0:
        self.start_point = self.labeled_idxs[0]
      else:
        self.start_point = [eid for eid in self.labeled_idxs]
      # Finding the next message to show -- get the current max element
      uidx = np.argmax(self.f[self.unlabeled_idxs])
      self.next_message = self.unlabeled_idxs[uidx]
      # Now that a new message has been selected, mark it as unseen
      self.seen_next = False 

      self.iter = 0
      self.hits = [sum(init_labels.values())]   

    if self.params.verbose:
      print ("Done with the initialization.")
    
    self.initialized = True

  def setLabel (self, idx, lbl, display_iter = None):
    # Set label for given message id

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = 0 if lbl <= 0 else 1
  
    # First, some book-keeping
    # If setLabel is called without "firstMessage," then set start_point
    if self.start_point is None:
      self.start_point = idx
    self.iter += 1
    self.labels[idx] = lbl
    self.unlabeled_idxs.remove(idx)

    self.BDinv[idx,idx] = self.Dinv[idx]*self.l/(1+self.l)
    self.q[idx] = lbl*1/(1+self.l)
    I_A = np.eye(self.n) - self.BDinv.dot(self.A)

    self.f =  nlg.solve(I_A, self.q)

    # Some more book-keeping
    self.labeled_idxs.append(idx)
    if self.iter == 0:
      self.hits.append(lbl)
    else:
      self.hits.append(self.hits[-1] + lbl)

    # Finding the next message to show -- get the current max element
    uidx = np.argmax(self.f[self.unlabeled_idxs])
    self.next_message = self.unlabeled_idxs[uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      elapsed = time.time() - t1
      display_iter = display_iter if display_iter else self.iter
      print( 'Iter: %i, Selected: %i, Hits: %i, Time: %f' %
        (display_iter, self.labeled_idxs[-1], self.hits[-1], elapsed))


############################# IMPACT FACTOR:
class impactFactorAS(shariAS):
  ### DOES NOT SCALE #########
  ### Assuming dense #########
  
  def __init__ (self, params=Parameters()):
    shariAS.__init__ (self, params)
    self.Minv = None

  def initialize(self, A, init_labels = {}):
    """
    A       --> n x n affinity matrix of feature values for each point.
    init_labels --> dictionary from index to label of initial labels.
    """
    # Save Xf and initialize some of the variables which depend on Xf
    self.A = A
    self.n = A.shape[0]

    self.labeled_idxs = init_labels.keys()
    self.unlabeled_idxs = list(set(range(self.n)) - set(self.labeled_idxs))

    self.labels = np.array([-1]*self.n)
    self.labels[self.labeled_idxs] = init_labels.values()

    # Initialize some parameters and constants which are needed and not yet initialized
    self.l = (1-self.params.eta)/self.params.eta
    if self.params.w0 is None:
      self.params.w0 = 1/self.n

    # Set up some of the initial values of some matrices
    self.D = matrix_squeeze(self.A.sum(1)) ##
    P = self.D * np.where(self.labels==-1, self.params.w0, 1./self.l)
    if self.params.remove_self_degree:
      if self.params.verbose:
        print("Removing diagonal elements.")
      self.D -= A.diagonal()
      self.A = self.A - np.diag(matrix_squeeze(A.diagonal()))

    if self.Minv is None:
      M = np.diag(self.D + P) - self.A

      if self.params.verbose:
        print ("Inverting M")
        t1 = time.time()

      self.Minv = np.matrix(nlg.inv(M))

      if self.params.verbose:
        print("Time for inverse:", time.time() - t1)

    self.q = P * np.where(self.labels==-1, self.params.pi, self.labels)
    
    self.f = matrix_squeeze(self.Minv.dot(self.q))

    # some constants for impact factor calculation
    # refer notes
    self.gamma1 = 1 / self.l - self.params.pi * self.params.w0
    self.gamma2 = 1 / self.l - self.params.w0
    if self.params.alpha > 0:
      self._computeImpactFactor()
    else:
      self.IM = 0
    
    # Setting iter/start_point
    # If batch initialization is done, then start_point is everything given
    if len(self.labeled_idxs) > 0:
      if len(self.labeled_idxs) == 0:
        self.start_point = self.labeled_idxs[0]
      else:
        self.start_point = [eid for eid in self.labeled_idxs]

      # Finding the next message to show -- get the current max element
      if self.params.alpha > 0:
        uidx = np.argmax((0*self.f + self.params.alpha * self.IM)[self.unlabeled_idxs])
      else:
        uidx = np.argmax((self.f))
      self.next_message = self.unlabeled_idxs[uidx]
      # Now that a new message has been selected, mark it as unseen
      self.seen_next = False 

      self.iter = 0
      self.hits = [sum(init_labels.values())]
    
    if self.params.verbose:
      print ("Done with the initialization.")
    
    self.initialized = True

  def _computeImpactFactor(self):
    """Internal function to compute impact factor."""
    DMinv_diag = self.D * np.diag(self.Minv)
    F_i = (self.f + self.gamma1 * DMinv_diag) / (1 + self.gamma2 * DMinv_diag)
    coeffs = self.D * (self.gamma1 - self.gamma2 * F_i)
    F = self.f.reshape(-1, 1) + self.Minv.A * coeffs
    u = np.where(self.labels == -1, 1., 0.)

    # Linear impact factor as described in Shari's paper
    if self.quad_IM:
      # Quadratic impact factor -- approximation of entropy
      E = F - F ** 2
      self.IM = - matrix_squeeze(self.f * (E.T.dot(u) - (F_i - F_i ** 2)))
      # print('Means quad: ', self.f.mean(), self.IM.mean())
    else:
      self.IM = matrix_squeeze(self.f * (F.T.dot(u) - (F_i - self.f) - self.f.dot(u)))
      # print('Means lin: ', self.f.mean(), self.IM.mean())

    self.IM *= (self.f.mean() / self.IM.mean())

  def setLabel (self, idx, lbl, display_iter = None):
    # Set label for given message id

    if self.params.verbose:
      t1 = time.time()

    # just in case, force lbl to be 0 or 1
    lbl = 0 if lbl <= 0 else 1
  
    # First, some book-keeping
    # If setLabel is called without "firstMessage," then set start_point
    if self.start_point is None:
      self.start_point = idx
    self.iter += 1
    self.labels[idx] = lbl
    self.unlabeled_idxs.remove(idx)

    Minv_i = self.Minv[:,idx]
    gamma = self.gamma2 / (1 / self.D[idx] + self.gamma2 * Minv_i[idx, 0])

    self.Minv -= gamma * Minv_i.dot(Minv_i.T)

    self.q[idx] = self.D[idx] * lbl / self.l
  
    self.f = matrix_squeeze(self.Minv.dot(self.q))

    if self.params.alpha > 0:
      self._computeImpactFactor()

    # Some more book-keeping
    # Not sure how we're going to change "history" in some sense:
    # -- everything since this was first incorrectly labeled.
    self.labeled_idxs.append(idx)
    if self.iter == 0:
      self.hits.append(lbl)
    else:
      self.hits.append(self.hits[-1] + lbl)

    # Finding the next message to show -- get the current max element
    if self.params.alpha > 0:
      uidx = np.argmax((0 * self.f + self.params.alpha * self.IM)[self.unlabeled_idxs])
    else:
      uidx = np.argmax(self.f[self.unlabeled_idxs])
    self.next_message = self.unlabeled_idxs[uidx]
    # Now that a new message has been selected, mark it as unseen
    self.seen_next = False 

    if self.params.verbose:
      elapsed = time.time() - t1
      display_iter = display_iter if display_iter else self.iter
      print( 'Iter: %i, Selected: %i, Hits: %i, Time: %f' % 
        (display_iter, self.labeled_idxs[-1], self.hits[-1], elapsed))