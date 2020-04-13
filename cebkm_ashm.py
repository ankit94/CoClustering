import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import inv
from datasets import prepare_dataset


def get_one_hot(indices, n_rows, n_cols):
    new_arr = np.zeros((n_rows, n_cols))
    for index,val in enumerate(indices):
            new_arr[index, val] = 1
    return new_arr

def prepare_r_matrix(data_matrix, n_clusters, n_iterations):
    len_data_matrix = len(data_matrix)
    R = np.array([])

    #Run kmeans m times
    for i in range(n_iterations):
        kmeans_res = KMeans(n_clusters=3).fit(data_matrix).labels_

        #Prepare the one hot representation
        new_arr = get_one_hot(kmeans_res, len_data_matrix, n_clusters)

        #Update the R matrix
        if i == 0:
            R = new_arr
        else:
            R = np.concatenate((R, new_arr), axis = 1)
    return R

def initialize_f_and_g(data_matrix, n_clusters, method = None):
    data_matrix = np.array(data_matrix)
    n_rows, n_cols = data_matrix.shape
    F = get_one_hot(KMeans(n_clusters).fit(data_matrix).labels_, n_rows, n_clusters)
    G = get_one_hot(KMeans(n_clusters).fit(data_matrix.transpose()).labels_, n_cols, n_clusters)
    return F, G


def do_until_converge(R, F, G):
    prev = F
    while prev != G:
        term_one = inv(np.matmul(np.matmul(F.T, F), np.matmul(G.T, G)).T)
        term_two = np.matmul(F.T, np.matmul(R, G))
        S = np.matmul(term_one, term_two)
        W = np.matmul(F, S)
        # TBD


if __name__ == "__main__":
    #Read and process data
    data_matrix, ground_truth = prepare_dataset(2)

    # Step 1 : Prepare R : Dont know if we need to convert R to a bipartite graph
    R = prepare_r_matrix(data_matrix, 3, 2816)
    np.savetxt("r.csv", R, delimiter=",")

    # Step 2 : Initialize F and G, the indicator matrices
    F, G = initialize_f_and_g(data_matrix, 3)

    # Step 3 : Converge now
    final_answer = do_until_converge(R, F, G)
