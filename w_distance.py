import numpy as np
import ot
from scipy.optimize import linprog
from scipy.spatial.distance import squareform, pdist
from scipy import stats

def w_distance(x, p, q):
    '''
    Compute the 1-Wasserstein distance between two probability distributions with POT package.
    Input:
        x: ndarray, indicates location of the bins, has shape (num_bins, num_dimensions)
        p: ndarray, indicates the probability of each bin, has shape (num_bins,)
        q: ndarray, indicates the probability of each bin, has shape (num_bins,)

    Output: 
        distance: the 1-Wasserstein distance between p and q
        T: ndarray, has shape (num_bins, num_bins), the optimal transport matrix
    '''
    # if data is 1D, use scipy.stats library to compute the 1-Wasserstein distance
    if x.ndim == 1:
        return stats.wasserstein_distance(x, x, p, q)
    # check the input
    assert x.shape[0] == p.shape[0] == q.shape[0], 'The number of bins should be the same for p and q'
    assert np.allclose(np.sum(p), 1), 'The sum of p should be 1'
    assert np.allclose(np.sum(q), 1), 'The sum of q should be 1'

    # compute loss matrix
    M = squareform(pdist(x, metric="sqeuclidean"))

    # compute the 1-Wasserstein distance
    T = ot.emd(p, q, M)
    distance = np.sum(T * M)

    return distance, T

if __name__ == '__main__':
    # create two discrete distributions for testing
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    p = np.array([0.25, 0.25, 0.25, 0.25])
    q = np.array([0.25, 0.25, 0.25, 0.25])



    # compute the 1-Wasserstein distance
    distance, T = w_distance(x, p, q)
    print('The 1-Wasserstein distance between x and y is {}'.format(distance))
