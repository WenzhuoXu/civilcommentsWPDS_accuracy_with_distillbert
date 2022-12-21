from datasets.civilcomments_wpds import CivilCommentsWPDS
import numpy as np
from typing import Any, Iterable, Optional
import torch
from pulp import *
import pandas as pd
import matplotlib.pyplot as plt
from w_distance import *

def to_ndarray(item: Any, dtype: np.dtype = None) -> np.ndarray:
    r"""
    Overview:
        Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.
    Arguments:
        - item (:obj:`object`): the item to be changed
        - dtype (:obj:`type`): the type of wanted ndarray
    Returns:
        - item (:obj:`object`): the changed ndarray
    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """
    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if item.device != 'cpu':
            item = item.detach().cpu()
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))

def get_data_1(dir='datasets/civilcomments', batch_num=0, alpha=0.1):
    """
        prob_1: prob dis of batch; [2, dim]
        prob_2: prob dis of last $alpha$ porpotion of batch [2, dim]
    """

    dataset = CivilCommentsWPDS(root_dir=dir, download=False)
    batch_0 = dataset.get_batch(t=batch_num)
    # print(dir(batch_0))
    # print(batch_0.metadata_array)
    y_array = to_ndarray(batch_0.y_array)
    metadata_array = to_ndarray(batch_0.metadata_array)
    
    # prob_1 --> prob dis of batch ; prob_2 --> prob dis of last $alpha$ porpotion of batch 
    prob_1, prob_2 = np.zeros(metadata_array.shape[1]), np.zeros(metadata_array.shape[1])
    batch_size = metadata_array.shape[0]
    batch_index = int(batch_size * alpha)
    for i in range(metadata_array.shape[1]):
        prob_1[i] = metadata_array[:, i].sum() / len(metadata_array[:, i])
        prob_2[i] = metadata_array[-batch_index:, i].sum() / batch_index

    # convert the prob into standard multinomial
    prob_1_m = []
    prob_2_m = []

    for i in range(len(prob_1)):
        p_1 = prob_1[i]
        p_2 = prob_2[i]
        prob_1_m.append([p_1, 1-p_1])
        prob_2_m.append([p_2, 1-p_2])

    prob_1_m = np.array(prob_1_m)
    prob_2_m = np.array(prob_2_m)

    return prob_1_m, prob_2_m

def get_data_2(dir='datasets/civilcomments', batch_num=0, beta=0.7, alpha=0.1):
    """
        prob_1_m: prob dis of $beta$ porpotion of batch; [2, dim]
        prob_2_m: prob dis from $beta-alpha$ to $beta$ of batch [2, dim]

        In this case, the prob is multinomial
    """

    dataset = CivilCommentsWPDS(root_dir=dir, download=False)
    batch_0 = dataset.get_batch(t=batch_num)
    # print(dir(batch_0))
    # print(batch_0.metadata_array)
    y_array = to_ndarray(batch_0.y_array)
    metadata_array = to_ndarray(batch_0.metadata_array)
    
    # prob_1 --> prob dis of batch ; prob_2 --> prob dis of last $alpha$ porpotion of batch 
    prob_1, prob_2 = np.zeros(metadata_array.shape[1]), np.zeros(metadata_array.shape[1])
    batch_size = metadata_array.shape[0]
    batch_index_beta = int(batch_size * beta)
    batch_index_alpha = int(batch_size * (beta+alpha))
    for i in range(metadata_array.shape[1]):
        prob_1[i] = metadata_array[:batch_index_beta, i].sum() / batch_index_beta
        prob_2[i] = metadata_array[batch_index_beta:batch_index_alpha, i].sum() / (batch_index_alpha - batch_index_beta)

    # convert the prob into standard multinomial
    prob_1_m = []
    prob_2_m = []

    for i in range(len(prob_1)):
        p_1 = prob_1[i]
        p_2 = prob_2[i]
        prob_1_m.append([p_1, 1-p_1])
        prob_2_m.append([p_2, 1-p_2])

    prob_1_m = np.array(prob_1_m)
    prob_2_m = np.array(prob_2_m)

    return prob_1_m, prob_2_m

def get_data_3(dir='datasets/civilcomments', batch_num=0) -> np.ndarray:
    """
        prob: the prob dist of one batch $batch_num$
    """
    dataset = CivilCommentsWPDS(root_dir=dir, download=False)
    batch_data = dataset.get_batch(t= batch_num)

    y_array = to_ndarray(batch_0.y_array)
    metadata_array = to_ndarray(batch_0.metadata_array)

    prob = np.zeros(metadata_array.shape[1])

    for i in range(metadata_array.shape[1]):
        prob[i] = metadata_array[:, i].sum() / metadata_array.shape[0]

    # convert the prob into standard multinomial
    prob_m = []
    for i in range(len(prob)):
        p = prob[i]
        prob_m.append([p, 1-p])

    return prob_m

def epsilon_kl_divergence(y_recent, y_average, epsilon=0.02):
    # This function takes two probability distributions as input, and outputs its kl divergence. 
    # For a discrete distribution the divergence will be computed
    # exactly as is described in Runtian's paper.
    ind_recent = len(y_recent)
    ind_ave = len(y_average)

    if(ind_recent != ind_ave):
        print('The source and target data must have the same labels.') 

    div_label = np.zeros(ind_recent) # initialize divergence by labels
    for i in range(ind_recent):
        div_label[i] = np.sum(y_recent[i] * y_average[i] / \
        np.maximum(y_recent[i], epsilon)) + np.sum(y_average[i] * np.log(y_average[i] / \
        np.maximum(y_average[i], epsilon) + 1))

    # The total divergence can be seen as a joint distribution of 
    # separate divergences. Assume the label probabilities are the same.

    return np.sum(div_label)

def construct_c(X_p, X_q):
    m = len(X_p)
    n = len(X_q)
    
    Y = [1/m]*m + [-1/n]*n
    
    return np.asarray(Y)


def rho(x,y):
    return abs(x-y)


def construct_b(X_p, X_q):
    
    X = np.concatenate((X_p, X_q), axis=0)
    N = len(X)
    
    b_part = []
    for i in range(N):
        for j in range(i+1, N):
            b_part.append(rho(X[i], X[j]))
    
    # Now, we duplicate each row to obtain a list of size 2*N
    b = []
    for i in range(N):
        b.append(b_part[i])
        b.append(b_part[i])
    
    return b


def construct_M(X_p, X_q):
    
    X = np.concatenate((X_p, X_q), axis=0)
    N = len(X)
    
    M = []
    for i in range(N):
        for j in range(i+1, N):
            l_M_1 = [0]*N
            l_M_1[i] = 1
            l_M_1[j] = -1
            M.append(l_M_1)
            l_M_2 = [0]*N
            l_M_2[i] = -1
            l_M_2[j] = 1
            M.append(l_M_2)
    M = np.asarray(M)

    return M.astype(int)

def kantorovich_metric(X_p, X_q):
    
    X = np.concatenate((X_p, X_q), axis=0)
    m = len(X_p)
    n = len(X_q)
    N = m+n
    
    c = construct_c(X_p, X_q)
    b = construct_b(X_p, X_q)
    M = construct_M(X_p, X_q)
    
    prob = LpProblem("LP_problem_for_estimating_the_Kantorovich_metric", LpMaximize)
    a = LpVariable.matrix("a", list(range(N)))
    prob += lpDot(c, a)
    p = 2*N
    for i in range(p):
        prob += lpDot(M[i], a) <= b[i]
    prob.solve()
    
#     for v in prob.variables():
#         print(v.name, "=", v.varValue)
        
#     print("objective=", value(prob.objective))

    return value(prob.objective)

def plot_epsilon_KL(dir='datasets/civilcomments', batch_num=0, alpha=0.02):
    epsilon_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    for epsilon in epsilon_list:
        print("epsilon: "+str(epsilon))
        beta_list = np.linspace(0.1, 1-alpha, 40)
        div = []
        for beta in beta_list:
            print("beta: "+str(beta))
            prob_1_i, prob_2_i = get_data_2(dir='datasets/civilcomments', batch_num=batch_num, beta=beta, alpha=alpha)
            div_i = epsilon_kl_divergence(y_recent=prob_2_i, y_average=prob_1_i, epsilon=epsilon)
            div.append(div_i)
        div = np.array(div)
        plt.plot(beta_list, div, label="{}-KL divergence".format(epsilon))

    plt.xlabel("beta")
    plt.ylabel("epsilon-KL divergence")
    plt.legend(loc='lower center')
    plt.savefig("figures/epsilon-KL_batch_{}_alpha_{}.png".format(batch_num, alpha))
    
def plot_W(dir='datasets/civilcomments', batch_num=0, alpha=0.02):
    epsilon_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    # for epsilon in epsilon_list:
    #     print("epsilon: "+str(epsilon))
    
    beta_list = np.linspace(0.1, 1-alpha, 40)
    div = []
    for beta in beta_list:
        print("beta: "+str(beta))
        prob_1_i, prob_2_i = get_data_2(dir='datasets/civilcomments', batch_num=batch_num, beta=beta, alpha=alpha)
        div_i = w_distance(x=prob_2_i, y=prob_1_i)
        div.append(div_i)
    div = np.array(div)
    plt.plot(beta_list, div, label="W divergence")

    plt.xlabel("beta")
    plt.ylabel("epsilon-KL divergence")
    plt.legend(loc='lower center')
    plt.savefig("figures/epsilon-KL_batch_{}_alpha_{}.png".format(batch_num, alpha))

def plot_epsilon_KL_mulbatch(dir='datasets/civilcomments', epsilon=0.2, alpha=0.02):
    return
    print("epsilon: "+str(epsilon))
    beta_list = np.linspace(0.1, 16-alpha, 100)
    div = []
    for beta in beta_list:
        print("beta: "+str(beta))
        batch_num = int(beta)
        beta = beta - batch_num
        prob_1_i, prob_2_i = get_data_2(dir='datasets/civilcomments', batch_num=batch_num, beta=beta, alpha=alpha)
        div_i = epsilon_kl_divergence(y_recent=prob_2_i, y_average=prob_1_i, epsilon=epsilon)
        div.append(div_i)
    div = np.array(div)
    plt.figure(figsize=(16, 4))
    plt.plot(beta_list, div, label="{}-KL divergence".format(epsilon))

    plt.xlabel("beta")
    plt.ylabel("epsilon-KL divergence")
    plt.legend()
    plt.savefig("figures/whole_data_{}-KL_alpha_{}.png".format(epsilon, alpha))

def cal_prob_1_batch(dir='datasets/civilcomments'):
    dataset = CivilCommentsWPDS(root_dir=dir, download=False)
    batch_0 = dataset.get_batch(t=batch_num)
    # print(dir(batch_0))
    # print(batch_0.metadata_array)
    y_array = to_ndarray(batch_0.y_array)
    metadata_array = to_ndarray(batch_0.metadata_array)
    
    # prob_1 --> prob dis of batch ; prob_2 --> prob dis of last $alpha$ porpotion of batch 
    prob_1, prob_2 = np.zeros(metadata_array.shape[1]), np.zeros(metadata_array.shape[1])
    batch_size = metadata_array.shape[0]
    batch_index = int(batch_size * alpha)
    for i in range(metadata_array.shape[1]):
        prob_1[i] = metadata_array[:, i].sum() / len(metadata_array[:, i])
        prob_2[i] = metadata_array[-batch_index:, i].sum() / batch_index

    # convert the prob into standard multinomial
    prob_1_m = []
    prob_2_m = []

    for i in range(len(prob_1)):
        p_1 = prob_1[i]
        p_2 = prob_2[i]
        prob_1_m.append([p_1, 1-p_1])
        prob_2_m.append([p_2, 1-p_2])

    prob_1_m = np.array(prob_1_m)
    prob_2_m = np.array(prob_2_m)
        
if __name__ == "__main__":
    for i in range(1, 10):    
        plot_W(dir='datasets/civilcomments', batch_num=0, alpha=0.01*i)


    