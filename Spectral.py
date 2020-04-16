import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import normalized_mutual_info_score
from datasets import get_data_set
import argparse

class Spectral(object):
    def __init__(self, dataset):
        data_map = {
            'classic3': 1,
            'mnist': 2,
            'cstr': 3
        }
        self.dataset = dataset
        print("Fetching ", dataset)
        self.data, self.labels = get_data_set(data_map[dataset])

    def view_dataset(self, title, data, markersize=0.001):
        plt.spy(data, markersize=markersize)
        plt.title(title)
        plt.show()

    def shuffle_data(self):
        print("Shuffling")
        self.data, self.labels = shuffle(self.data, self.labels)
        self.view_dataset(data=self.data, title='shuffled data')

    def form_biclusters(self):
        print("Generating clusters")
        self.bicluster = SpectralCoclustering(
            n_clusters=len(np.unique(self.labels)),
            n_jobs=-1)
        self.bicluster.fit(self.data)

    def get_accuracy(self):
        nmi = normalized_mutual_info_score(
            self.bicluster.row_labels_,
            self.labels)
        print("Accuracy is ", nmi)

    def show_clusters(self):
        fit_data = self.data[np.argsort(self.bicluster.row_labels_)]
        fit_data = fit_data[:, np.argsort(self.bicluster.column_labels_)]
        self.view_dataset(data=fit_data, title='co-clusters')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=['classic3', 'cstr', 'mnist'],
        required=True) 
    args = parser.parse_args()

    cocluster = Spectral(args.dataset)
    cocluster.shuffle_data()
    cocluster.form_biclusters()
    cocluster.get_accuracy()
    cocluster.show_clusters()
