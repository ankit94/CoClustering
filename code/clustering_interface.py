import subspace
import spectral
import infotheoretic
import argparse
import sys, os
sys.path.append(os.path.abspath('../'))

clustering_technique = {
    'spectral':spectral.perform_clustering,
    'subspace': subspace.perform_clustering,
    'infoth':infotheoretic.perform_clustering
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--clustering_technique",
        choices=['spectral', 'subspace', 'infoth'],
        required=True)

    parser.add_argument(
        "-d",
        "--dataset",
        choices=['classic3', 'cstr', 'mnist'],
        required=True)

    args = parser.parse_args()
    clustering_technique[args.clustering_technique](args.dataset)
