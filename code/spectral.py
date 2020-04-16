import sys, os
sys.path.append(os.path.abspath('../'))

import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import normalized_mutual_info_score
from data.datasets import get_data_set


class Spectral(object):
    def __init__(self, dataset):
        """initialize spectral class
        
        Arguments:
            dataset {np.array} -- dataset to fetch
        """
        data_map = {
            'classic3': 1,
            'cstr': 3,
            'mnist': 2
        }
        self.dataset = dataset
        print("Fetching ", dataset)
        self.data, self.labels = get_data_set(data_map[dataset])
        if (~self.data.any(axis=0)).any():
            print("Found empty features. deleting...")
            self.data = np.delete(
                self.data,
                np.where(~self.data.any(axis=0))[0],
                axis=1)

    def view_dataset(self, title, data, markersize=0.001):
        """plot data matrix
        
        Arguments:
            title {str} -- title of plot
            data {np.array} -- dataset to plot
        
        Keyword Arguments:
            markersize {float} -- size of datapoints (default: {0.001})
        """
        plt.spy(data, markersize=markersize)
        plt.title(title)
        plt.show()

    def shuffle_data(self):
        """shuffles self.data
        """
        print("Shuffling")
        self.data, self.labels = shuffle(self.data, self.labels)
        self.view_dataset(data=self.data, title='shuffled data')

    def form_biclusters(self):
        """generates spectral bi-clusters from self.data
        """
        n_clusters = len(np.unique(self.labels))
        print("Generating {} clusters".format(n_clusters))
        self.bicluster = SpectralCoclustering(
            n_clusters=n_clusters,
            n_jobs=-1)
        self.bicluster.fit(self.data)

    def get_accuracy(self):
        """calculates NMI between self.bicluster rows and data labels
        """
        nmi = normalized_mutual_info_score(
            self.bicluster.row_labels_,
            self.labels)
        print("Accuracy is ", nmi)

    def show_clusters(self):
        """sorts data according to bicluster row and col labels and plots
        """
        fit_data = self.data[np.argsort(self.bicluster.row_labels_)]
        fit_data = fit_data[:, np.argsort(self.bicluster.column_labels_)]
        self.view_dataset(data=fit_data, title='co-clusters')

def perform_clustering(dataset):
    cocluster = Spectral(dataset)
    cocluster.shuffle_data()
    cocluster.form_biclusters()
    cocluster.get_accuracy()
    cocluster.show_clusters()
