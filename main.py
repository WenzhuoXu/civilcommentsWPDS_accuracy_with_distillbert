from datasets.civilcomments_wpds import CivilCommentsWPDS
import numpy as np
from typing import Any, Iterable, Optional
import torch
from pulp import *
import pandas as pd
import matplotlib.pyplot as plt
import ot
from scipy.optimize import linprog
from scipy.spatial.distance import squareform, pdist
from scipy import stats


def Gaussian_data_generation(mu = [0, 0], cov = [[1, 0], [0, 100]], num = 30, threshold = 10):
    # threshold = 10
    mag = 2
    
    X = np.random.multivariate_normal(mu, cov, num).T
    
    X = np.array(X) * mag
    X = (X+0.5).astype('int32')

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j] > threshold:
                X[i][j] = threshold
            if X[i][j] < -1 * threshold:
                X[i][j] = -1 * threshold
    
    prob_alldim = []
    low = -1 * threshold
    high = threshold
    
    
    for i in range(X.shape[0]): # dim
        data = X[i] 
        prob_dim_i = np.zeros(high-low+1) # bin
        for j in range(X.shape[1]):
            num = data[j]
            prob_dim_i[num-low] += 1
        prob_dim_i = prob_dim_i / X.shape[1]
        prob_alldim.append(prob_dim_i)
        
    prob_alldim = np.array(prob_alldim)
    
    return prob_alldim

def Gaussian_data_generation_X(mu = [0, 0], cov = [[1, 0], [0, 100]], num = 30):
    threshold = 10
    mag = 1
    
    X = np.random.multivariate_normal(mu, cov, num).T
    
    X = np.array(X) * mag
    X = (X+0.5).astype('int32')

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j] > threshold:
                X[i][j] = threshold
            if X[i][j] < -1 * threshold:
                X[i][j] = -1 * threshold
    
    prob_alldim = []
    low = -1 * threshold
    high = threshold
    
    
    for i in range(X.shape[0]): # dim
        data = X[i] 
        prob_dim_i = np.zeros(high-low+1) # bin
        for j in range(X.shape[1]):
            num = data[j]
            prob_dim_i[num-low] += 1
        prob_dim_i = prob_dim_i / X.shape[1]
        prob_alldim.append(prob_dim_i)
        
    prob_alldim = np.array(prob_alldim)
    
    return prob_alldim, X

def Gaussian_data_generation_P(X, threshold):
    low = -1 * threshold
    high = threshold
    prob_alldim = []
    for i in range(X.shape[0]): # dim
        data = X[i] 
        prob_dim_i = np.zeros(high-low+1) # bin
        for j in range(X.shape[1]):
            num = data[j]
            prob_dim_i[num-low] += 1
        prob_dim_i = prob_dim_i / X.shape[1]
        prob_alldim.append(prob_dim_i)
        
    return prob_alldim

def Data_permutation(X, Y):

    # permutate X, Y randomly
    
    total_data = np.hstack((X, Y))
    indices = np.arange(total_data.shape[1])
    np.random.shuffle(indices)
    total_data_per = total_data[:, indices]
    
    index_1 = X.shape[1]
    index_2 = Y.shape[1]
    
    X_per = total_data_per[:, :index_1]
    Y_per = total_data_per[:, -1*index_2: ]
    
    return X_per, Y_per

def Permutation_test(mu_t = 0.5, per_num = 1000):
    threshold = 10
    prob_X, X = Gaussian_data_generation_X(mu = [0], cov = [[5]], num = 5000)
    prob_Y, Y = Gaussian_data_generation_X(mu = [mu_t], cov = [[5]], num = 100)
    t = epsilon_kl_divergence(prob_Y, prob_X)
    # t = np.abs(X.mean() - Y.mean())
    
    p_value = 0
    
    for _ in range(per_num):
        X_per, Y_per = Data_permutation(X, Y)
        prob_X_per = Gaussian_data_generation_P(X_per, threshold)
        prob_Y_per = Gaussian_data_generation_P(Y_per, threshold)
        t_i = epsilon_kl_divergence(prob_Y_per, prob_X_per)
        # t_i = np.abs(X_per.mean() - Y_per.mean())
        # print("t: {}; t_i: {}".format(t, t_i))
        if (t_i > t):
            p_value += 1
    
    p_value = p_value / per_num
    
    return p_value

def Permutation_test_W(mu_t = 0.5, per_num = 1000):
    threshold = 10
    prob_0, X = Gaussian_data_generation_X(mu = [0], cov = [[2]], num = 5000)
    prob_1, Y = Gaussian_data_generation_X(mu = [mu_t], cov = [[2]], num = 100)
    
    x = np.linspace(-1 * threshold, threshold, 2*threshold+1)

    prob_0 = prob_0.reshape(-1)
    prob_1 = prob_1.reshape(-1)

    t, T = w_distance(x, prob_0, prob_1)
    
    # t = np.abs(X.mean() - Y.mean())
    
    p_value = 0
    
    for _ in range(per_num):
        X_per, Y_per = Data_permutation(X, Y)
        prob_X_per = Gaussian_data_generation_P(X_per, threshold)
        prob_Y_per = Gaussian_data_generation_P(Y_per, threshold)
        
        prob_X_per = np.array(prob_X_per)
        prob_Y_per = np.array(prob_Y_per)
        
        prob_X_per = prob_X_per.reshape(-1)
        prob_Y_per = prob_Y_per.reshape(-1)
        
        t_i, aaa = w_distance(x, prob_X_per, prob_Y_per)
        # t_i = np.abs(X_per.mean() - Y_per.mean())
        # print("t: {}; t_i: {}".format(t, t_i))
        if (t_i > t):
            p_value += 1
    
    p_value = p_value / per_num
    
    return p_value


def Permutation_test_list(per_num = 1000):
    mu_t_l = [0.1, 0.2, 0.3, 0.4, 0.5]
    for mu_i in mu_t_l:
        p_i = Permutation_test(mu_t = mu_i, per_num = per_num)
        print("mu: {}, p: {}".format(mu_i, p_i))
        
def Permutation_test_list_W(per_num = 1000):
    mu_t_l = [0.1, 0.2, 0.3, 0.4, 0.5]
    for mu_i in mu_t_l:
        p_i = Permutation_test_W(mu_t = mu_i, per_num = per_num)
        print("mu: {}, p: {}".format(mu_i, p_i))
    
    
def GetMixtureDis(mu_1=[0, 0], cov_1=[[1, 0], [0, 1]], mu_2= [3, 3], cov_2=[[1, 0], [0, 10]], gamma = 0.1, n = 1000, threshold = 10):
    prob1 = Gaussian_data_generation(mu=mu_1, cov=cov_1, num=n, threshold=threshold)
    prob2 = Gaussian_data_generation(mu=mu_2, cov=cov_2, num=n, threshold=threshold)
    
    # if gamma == 0:
    #     return prob1
    # if gamma == 1:
    #     return prob2
    
    prob = np.copy(prob1)
    for i in range(len(prob1)):
        prob[i] = prob1[i] * (1-gamma) + prob2[i] * gamma
    
    return prob1, prob

def epsilon_kl_divergence(y_recent, y_average, epsilon=0.2):
    # This function takes two probability distributions as input, and outputs its kl divergence. 
    # For a discrete distribution the divergence will be computed
    # exactly as is described in Runtian's paper.
    ind_recent = len(y_recent)
    ind_ave = len(y_average)

    if(ind_recent != ind_ave):
        print('The source and target data must have the same labels.') 

    div_label = np.zeros(ind_recent) # initialize divergence by labels
    for i in range(ind_recent):
        div_label[i] = np.sum(-1 * y_recent[i] * y_average[i] / \
        np.maximum(y_recent[i], epsilon)) + np.sum(y_average[i] * np.log(np.maximum(y_average[i], epsilon) / \
        np.maximum(y_recent[i], epsilon))) + 1

    # for i in range(ind_recent):
    #     div_label[i] = np.sum(y_recent[i] * y_average[i] / \
    #     np.maximum(y_recent[i], epsilon)) + np.sum(y_average[i] * np.log(y_average[i] / \
    #     np.maximum(y_average[i], epsilon))+ 1)  

    # The total divergence can be seen as a joint distribution of 
    # separate divergences. Assume the label probabilities are the same.

    return np.sum(div_label)

def plot_epsilon_KL(mu_1=[0, 0], cov_1=[[3, 0], [0, 3]], mu_2=[2, 2], cov_2=[[3, 0], [0, 3]]):
    epsilon_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    for epsilon in epsilon_list:
        print("epsilon: "+str(epsilon))
        gamma_list = np.linspace(0, 1, 40)
        div = []
        # prob_0 = GetMixtureDis(
        #     gamma=0,
        #     mu_1=mu_1,
        #     cov_1=cov_1
        # )
        
        for gamma in gamma_list:
            print("gamma: "+str(gamma))
            prob_0, prob_1 = GetMixtureDis(
                gamma=gamma,
                mu_1=mu_1,
                cov_1=cov_1,
                mu_2=mu_2,
                cov_2=cov_2
            )
            
            div_i = epsilon_kl_divergence(
                y_recent=prob_1, y_average=prob_0, epsilon=epsilon,
            )
            div.append(div_i)
            
        div = np.array(div)
        plt.plot(gamma_list, div, label="{}-KL divergence".format(epsilon))

    plt.xlabel("gamma")
    plt.ylabel("epsilon-KL divergence")
    plt.legend(loc='upper left')
    plt.savefig("figures/epsilon-KL_mu1_{}_mu2_{}_cov1_{}_cov2_{}.png".format(mu_1, mu_2, cov_1, cov_2))

def plot_W_dis(mu_1=[0], cov_1=[[1]], mu_2=[0], cov_2=[[10]]):
    threshold = 5
    gamma_list = np.linspace(0, 1, 40)
    div = []
    for gamma in gamma_list:
        # print("gamma: "+str(gamma))
    
        prob_0, prob_1 = GetMixtureDis(
                    gamma=gamma,
                    mu_1=mu_1,
                    cov_1=cov_1,
                    mu_2=mu_2,
                    cov_2=cov_2,
                    threshold=threshold
                )
    
        x = np.linspace(-1 * threshold, threshold, 2*threshold+1)

        prob_0 = prob_0.reshape(-1)
        prob_1 = prob_1.reshape(-1)
    
        distance, T = w_distance(x, prob_0, prob_1)
        div.append(distance)
        # print('The 1-Wasserstein distance between x and y is {}'.format(distance))
        
    plt.plot(gamma_list, div, label="1-Wasserstein distance")
    plt.xlabel("beta")
    plt.ylabel("1-Wasserstein distance")
    plt.savefig("figures/1-Wasserstein_mu1_{}_mu2_{}_cov1_{}_cov2_{}.png".format(mu_1, mu_2, cov_1, cov_2))
    
    
    # for epsilon in epsilon_list:
    #     print("epsilon: "+str(epsilon))
    #     gamma_list = np.linspace(0, 1, 40)
    #     div = []
        
        
        
    #     for gamma in gamma_list:
    #         print("gamma: "+str(gamma))
    #         prob_0, prob_1 = GetMixtureDis(
    #             gamma=gamma,
    #             mu_1=mu_1,
    #             cov_1=cov_1,
    #             mu_2=mu_2,
    #             cov_2=cov_2
    #         )
            
    #         div_i = epsilon_kl_divergence(
    #             y_recent=prob_1, y_average=prob_0, epsilon=epsilon,
    #         )
    #         div.append(div_i)
            
    #     div = np.array(div)
    #     plt.plot(gamma_list, div, label="{}-KL divergence".format(epsilon))

    # plt.xlabel("beta")
    # plt.ylabel("epsilon-KL divergence")
    # plt.legend(loc='upper left')
    # plt.savefig("figures/epsilon-KL_mu1_{}_mu2_{}_cov1_{}_cov2_{}.png".format(mu_1, mu_2, cov_1, cov_2))
    
def plot_single_epsilon_KL(mu_1=[0, 0], cov_1=[[1, 0], [0, 1]], mu_2=[2, 2], cov_2=[[5, 0], [0, 5]], rej_line = 0.22175):
    epsilon_list = [0.2]
    
    for epsilon in epsilon_list:
        print("epsilon: "+str(epsilon))
        gamma_list = np.linspace(0, 1, 40)
        div = []
        # prob_0 = GetMixtureDis(
        #     gamma=0,
        #     mu_1=mu_1,
        #     cov_1=cov_1
        # )
        
        for gamma in gamma_list:
            print("gamma: "+str(gamma))
            prob_0, prob_1 = GetMixtureDis(
                gamma=gamma,
                mu_1=mu_1,
                cov_1=cov_1,
                mu_2=mu_2,
                cov_2=cov_2
            )
            
            div_i = epsilon_kl_divergence(
                y_recent=prob_1, y_average=prob_0, epsilon=epsilon,
            )
            div.append(div_i)
            
        div = np.array(div)
        plt.plot(gamma_list, div, label="{}-KL divergence".format(epsilon))
        rej_thres = gamma_list* 0 + div[0] + rej_line
        plt.plot(gamma_list, rej_thres, label="rejection threshold")

    plt.xlabel("beta")
    plt.ylabel("epsilon-KL divergence")
    plt.legend(loc='upper left')
    plt.savefig("figures/single_epsilon-KL_mu1_{}_mu2_{}_cov1_{}_cov2_{}.png".format(mu_1, mu_2, cov_1, cov_2))


def test():
    prob1, prob2 = GetMixtureDis(gamma=0)
    div = epsilon_kl_divergence(y_recent=prob1, y_average=prob2, epsilon=0.001)
    print(div)
    
    
def Q1():
    X = [1.1, 2.3]
    X = int(X)
    print(X)
    
def Q2():
    x = [0.75156862, 0.35604343]
    x = np.array(x)
    
    x_int = (x + 0.5).astype('int32')
    
    X_norm = np.max(x_int, 2)
    print(X_norm)

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



if __name__ == "__main__":
    
    plot_epsilon_KL()
    
    # Fig 5
    plt.figure(1)
    plot_single_epsilon_KL(mu_1=[0], cov_1=[[1]], mu_2=[2], cov_2=[[10]], rej_line = 0.22175)
    
    # Fig 6
    plt.figure(2)
    plot_single_epsilon_KL(mu_1=[0], cov_1=[[1]], mu_2=[1], cov_2=[[8]], rej_line = 0.122)
    
    # Permutation test
    # Permutation_test_list()
    
    
    # plot_W_dis()
    
    # Permutation_test_list_W()
    
    # test()
    
    # test()
    # prob1 = GetMixtureDis(gamma = 0.2)
    # prob2 = GetMixtureDis(gamma = 0.8)
    # dis = epsilon_kl_divergence(prob1, prob2)
    
    # print(dis)

    # Gaussian_data_generation()
    # Q1()
    # Q2()