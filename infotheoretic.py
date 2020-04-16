from coclust.coclustering import CoclustInfo
from datasets import prepare_dataset
from sklearn.metrics import normalized_mutual_info_score as nmi
from coclust.visualization import plot_delta_kl, plot_convergence
import argparse

class InformationTheoretic:
    def __init__(self, dataset):
        """initialize InfoTh class

                Arguments:
                    dataset {np.array} -- dataset to fetch
                """
        data_map = {
            'classic3': [1,3],
            'mnist':[2,10],
            'cstr': [3,4]
        }
        self.dataset = dataset
        print("Fetching ", dataset)
        self.n_clusters = data_map[dataset][1]
        self.data_matrix, self.ground_truth = prepare_dataset(data_map[dataset][0])

    def perform_clustering(self):
        """Perform the Information theoretic clustering
            Arguments:
                data_matrix,
                number_of_clusters
        """
        model = CoclustInfo(n_row_clusters=self.n_clusters, n_col_clusters=self.n_clusters, n_init=4, random_state=0)
        model.fit(self.data_matrix)
        self.model = model
        self.predicted = model.row_labels_

    def evaluate_model(self):
        """calculates NMI
        Arguments:
            ground_truth,
            predicted_values
        """
        nmi_eval = nmi(self.ground_truth, self.predicted)
        print(f"NMI Accuracy is: {nmi_eval}")

    def plot_clusters(self):
        """plot clustering results

        Arguments:
            model
        """
        plot_convergence(self.model.criterions, 'P_KL MI')
        plot_delta_kl(self.model)

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=['classic3', 'mnist','cstr'],
        required=True)
    args = parser.parse_args()

    cocluster = InformationTheoretic(args.dataset)
    cocluster.perform_clustering()
    cocluster.evaluate_model()
    cocluster.plot_clusters()





