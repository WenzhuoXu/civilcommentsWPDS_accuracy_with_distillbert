import numpy as np
import ot
from scipy.optimize import linprog
from scipy.spatial.distance import squareform, pdist
from scipy import stats
from datasets.civilcomments_wpds import CivilCommentsWPDS
from civilcomments_utils import *
from civilcomments_train import *

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
        return stats.wasserstein_distance(x, x, p, q), None
    # check the input
    assert x.shape[0] == p.shape[0] == q.shape[0], 'The number of bins should be the same for p and q'
    # assert np.allclose(np.sum(p), 1), 'The sum of p should be 1'
    # assert np.allclose(np.sum(q), 1), 'The sum of q should be 1'

    # compute loss matrix
    M = squareform(pdist(x, metric="sqeuclidean"))

    # compute the 1-Wasserstein distance
    T = ot.emd(p, q, M)
    distance = np.sum(T * M)

    return distance, T

if __name__ == '__main__':
    # create two discrete distributions for testing
    x = np.array([1, 2, 3, 4])
    p = np.array([0.25, 0.25, 0.25, 0.25])
    q = np.array([0, 0.5, 0.5, 0])

    # compute the 1-Wasserstein distance
    distance, T = w_distance(x, q, p)
    print('The 1-Wasserstein distance between x and y is {}'.format(distance))
    '''
    # test 1-Wasserstein distance on CivilComments dataset
    # load the dataset
    model_name = 'distilbert-base-uncased'
    transform = initialize_bert_transform(model_name, 512)
    dataset1 = CivilCommentsWPDS(magic=2022).get_batch(t=0, transform=transform)
    dataset2 = CivilCommentsWPDS(magic=2022).get_batch(t=10, transform=transform)
    
    # transform into probability distributions
    prob1, prob2 = get_data_distribution(dataset1, dataset2)
    x = nn.functional.one_hot(dataset1.metadata_array).numpy()
    # compute the 1-Wasserstein distance
    # distance, T = w_distance(x, prob1, prob2)
    kl_div = epsilon_kl_divergence(prob1, prob2, epsilon=0.2)
    print(kl_div)
    '''

