from __future__ import division
import numpy as np

def get_activesearch_probs(similarity, labels, lam, pi, w0):
    '''
    Gets the vector of label probabilities for active search.
    
    Parameters
    ----------
    
    similarity: array, shape [n, n]
        The $A$ matrix of similarity scores.
    
    labels: array, shape n
        The labels for each node. 0 means negative, 1 means positive,
        negative value means unobserved.
    
    lam: nonnegative scalar
        Regularization parameter.
        
    pi: scalar in [0, 1]
        Prior probability of being positive.
        
    w0: nonnegative scalar
        Strength of that prior.
    '''
    D = similarity.sum(axis=1)
    n, = D.shape
    
    labeled = labels >= 0
    wts = np.where(labeled, lam / (1 + lam), 1 / (1 + w0))
    
    I_minus_Ap = (-wts / D)[:, None] * similarity
    I_minus_Ap[xrange(n), xrange(n)] += 1
    Dp_yp = (1 - wts) * np.where(labeled, labels, pi)
    return np.linalg.solve(I_minus_Ap, Dp_yp)
