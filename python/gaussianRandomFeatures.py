from __future__ import division, print_function
import cPickle
import numpy as np, numpy.random as nr, numpy.linalg as nlg


class GaussianRandomFeatures:
  """
  Class to store Gaussian Random Features.
  """
  def __init__(self, dim=None, rn=None, gammak=1.0, sine=False, fl=None):
    """
    Initialize with dim of input space, dim of random feature space
    and bandwidth of the RBF kernel.
    """
    if fl is not None:
      self.LoadFromFile(fl)
    else:
      self.dim = dim
      self.rn = rn
      self.gammak = gammak
      self.sine = sine

      self.generateCoefficients()

  def generateCoefficients (self):
    """
    Generate coefficients for GFF.
      """
    self.ws = []
    self.bs = []  # Unused if self.sine is True.
    mean = np.zeros(self.dim)
    cov = np.eye(self.dim)*(2*self.gammak)

    if self.sine:
      for _ in range(self.rn):
        self.ws.append(nr.multivariate_normal(mean, cov))
    else:
      for _ in range(self.rn):
        self.ws.append(nr.multivariate_normal(mean, cov))
        self.bs.append(nr.uniform(0.0, 2*np.pi))
      self.bs = np.array(self.bs)
    self.ws = np.array(self.ws)

  def computeRandomFeatures (self, f):
    """
    Projects onto fourier feature space.
    
    Assumes that f is n x r.
    """

    f = np.atleast_2d(f)
    if self.sine:
      rf_cos = np.cos(self.ws.dot(f.T))*np.sqrt(1/self.rn)
      rf_sin = np.sin(self.ws.dot(f.T))*np.sqrt(1/self.rn)
      return np.r_[rf_cos, rf_sin].T
    else:
      rf = np.cos(self.ws.dot(f.T) + self.bs[:, None])*np.sqrt(2/self.rn)
      return rf.T

  def RBFKernel(self, f1, f2, gammak=None):
    """
    Computes RBF Kernel.
    """
    if gammak is None: gammak = self.gammak
    
    f1 = np.array(f1)
    f2 = np.array(f2)

    return np.exp(-gammak*(nlg.norm(f1 - f2)**2))

  def LinearRandomKernel(self, f1, f2):
    """
    Computes Linear Kernel after projecting onto fourier space.
    """

    rf1 = self.computeRandomFeatures(f1)
    rf2 = self.computeRandomFeatures(f2)

    return np.squeeze(rf1).dot(np.squeeze(rf2))

  def SaveToFile(self, fl):
    data = {'dim': self.dim,
            'rn': self.rn,
            'gammak': self.gammak,
            'sine': self.sine,
            'ws': self.ws,
            'bs': self.bs}
    with open(fl, 'w') as fh:
      cPickle.dump(data, fh)

  def LoadFromFile(self, fl):
    with open(fl, 'r') as fh:
      data = cPickle.load(fh)
    self.dim = data['dim']
    self.rn = data['rn']
    self.gammak = data['gammak']
    self.sine = data['sine']

    self.ws = data['ws']
    self.bs = data['bs']


class RandomFeaturesConverter:

  def __init__(self, dim=None, rn=None, gammak=1.0, sine=False, feature_generator=None, fl=None):
    """
    dim   --> dimension of input space
    rn    --> number of random features
    gammak  --> bandwidth of rbf kernel
    sine  --> use sin in the random fourier features
    feature_generator  --> random feature generator
    fl --> File to save/load from.

    TODO: Why do we have this class again? Hmm.
    """
    self.dim = dim
    self.rn = rn
    self.gammak = gammak

    if feature_generator is None:
      self.feature_generator = GaussianRandomFeatures(self.dim, self.rn, self.gammak, sine=sine)
    else: self.feature_generator = feature_generator

  def getFeatureGenerator(self):
    """
    Get stored feature generator.
    """
    return self.feature_generator

  def getData (self, fs):
    """
    Gets the projected features.

    Data must be n x r.
    """
    return self.feature_generator.computeRandomFeatures(fs)

  def SaveToFile(self, fl):
    self.feature_generator.SaveToFile(fl)
