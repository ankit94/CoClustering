from coclust.coclustering import CoclustInfo
from datasets import prepare_dataset
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy as np
from coclust.evaluation.external import accuracy
from coclust.visualization import plot_delta_kl, plot_convergence
from sklearn.cluster import KMeans

def perform_clustering(data_matrix, n_clusters, method = "infoth"):
    model = CoclustInfo(n_row_clusters=n_clusters, n_col_clusters=n_clusters, n_init=4, random_state=0)
    model.fit(data_matrix)
    return model, np.array(model.row_labels_)

def evaluate_model(ground_truth, predicted):
    nmi_eval = nmi(ground_truth, predicted)
    print(f"NMI: {nmi_eval}")
    accuracy_eval = accuracy(ground_truth, predicted)
    print(f"ACCURACY: {accuracy_eval}")

def plot_clusters(model):
    plot_convergence(model.criterions, 'P_KL MI')
    plot_delta_kl(model)

if __name__  == "__main__":
    #Dataset 1
    n_clusters = 3
    data_matrix, ground_truth = prepare_dataset(1)
    model, predicted = perform_clustering(data_matrix, n_clusters)

    #Evaluate model
    evaluate_model(ground_truth, predicted)
    plot_clusters(model)

    #Dataset 2
    n_clusters2 = 10
    data_matrix2, ground_truth2 = prepare_dataset(2)
    model2, predicted2 = perform_clustering(data_matrix2, n_clusters2)

    # Evaluate model
    evaluate_model(ground_truth2, predicted2)
    plot_clusters(model2)





