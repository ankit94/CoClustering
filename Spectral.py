import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd

def view_dataset(data_matrix, title):
    plt.spy(data_matrix, markersize=0.001, )
    plt.title(title)
    plt.show()


def shuffle_data(data_matrix, document_names, feature_names):
    data_matrix, document_names = shuffle(data_matrix, document_names)
    data_matrix, feature_names = shuffle(data_matrix.transpose(), feature_names)
    return data_matrix.transpose() , document_names, feature_names


def form_biclusters(data_matrix, categories):
    bicluster = SpectralCoclustering(n_clusters=len(categories))    
    return bicluster.fit(data_matrix)

if __name__ == '__main__':
    df = pd.read_csv('classic3.csv')

    cats = df[' '].to_numpy()
    del df[' ']
    words = df.columns.to_numpy()

    
    data_matrix = df.to_numpy()
    view_dataset(data_matrix, "Original matrix")
    data_matrix, cats, words = shuffle_data(data_matrix, cats, words)
    view_dataset(data_matrix, "Shuffled matrix")

    bicluster = form_biclusters(data_matrix, np.unique(cats))
    t_labels = [1 if x == 'cisi' else (0 if x == 'med' else 2 ) for x in cats]
    nmi = normalized_mutual_info_score(bicluster.row_labels_, t_labels )
    print("Accuracy is {}".format(nmi))

    fit_data = data_matrix[np.argsort(bicluster.row_labels_)]
    fit_data = fit_data[:, np.argsort(bicluster.column_labels_)]

    view_dataset(fit_data, "Bi-clusters")