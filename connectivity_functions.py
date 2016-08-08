import numpy as np

epsilon = 1e-10

def get_w(P, p):

    w = np.zeros_like(P)
    aux = np.outer(p, p)
    probab = P / aux
    probab[probab < epsilon] = epsilon
    w = np.log(probab)

    return w

def get_beta(p):

    probability = p
    probability[p < epsilon] = epsilon

    beta = np.log(probability)

    return beta


def softmax(w, t=1.0):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    w : list of numbers
    t : float

    Return
    ------
    a list of the same length as w of non-negative numbers

    Examples
    --------
    >>> softmax([0.1, 0.2])
    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])
    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])
    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist