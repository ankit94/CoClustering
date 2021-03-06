import os
import numpy as np
from urllib.request import urlretrieve
from urllib.parse import urljoin
import random
import pandas as pd
from matplotlib import pyplot as plt
random.seed(1)
np.random.seed(1)
import scipy.io as io
import pathlib 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Dataset download path
mnist_url = 'http://yann.lecun.com/exdb/mnist/'
data_directory = pathlib.Path(__file__).parent


def mnist():
    file_descriptor = open(os.path.join(data_directory, 'train-images-idx3-ubyte'))
    data = np.fromfile(file=file_descriptor, dtype=np.uint8)
    data_matrix = data[16:].reshape((60000, 28*28)).astype(float)
    file_descriptor = open(os.path.join(data_directory, 'train-labels-idx1-ubyte'))
    data = np.fromfile(file=file_descriptor, dtype=np.uint8)
    ground_truth = data[8:].reshape((60000)).astype(float)

    #Scale the result and return
    min_max_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    data_matrix = std_scaler.fit_transform(data_matrix)
    data_matrix = min_max_scaler.fit_transform(data_matrix)
    return data_matrix, ground_truth

def get_classic_3():
    file = 'classic3.csv'
    data = pd.read_csv(os.path.join(data_directory, file))
    data.iloc[:, 0] = data.iloc[:, 0].replace('cran', int(0)).replace('cisi', int(1)).replace('med', int(2))
    data = data.to_numpy()
    ground_truth = data[:, 0]
    data_matrix = data[:, 1:]
    return data_matrix, ground_truth

def get_mnist():
    prepare_for_mnist()
    return mnist()

def get_cstr():
    file = 'cstr.mat'
    matlab_dict = io.loadmat(os.path.join(data_directory,file))
    data_matrix = np.array(matlab_dict['fea'])
    ground_truth = np.array(matlab_dict['gnd']).reshape(1,-1)[0]
    return data_matrix, ground_truth

def prepare_for_mnist():
    download_parse('train-images-idx3-ubyte.gz')
    download_parse('train-labels-idx1-ubyte.gz')

def download_parse(fgz):
    if os.path.exists(os.path.join(data_directory, fgz)):
        pass
    else:
        url = urljoin(mnist_url, fgz)
        filename = os.path.join(data_directory, fgz)
        urlretrieve(url, filename)
        os.system('gunzip ' + filename)

dataset  = {
    1 : get_classic_3,
    2 : get_mnist,
    3 : get_cstr
}

def prepare_dataset(dataset_number, **options):
    if not dataset_number:
        return []
    return get_data_set(dataset_number, **options)

def get_data_set(number, **options):
    return dataset[number](**options)







