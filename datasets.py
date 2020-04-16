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

from sklearn.decomposition import PCA


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Dataset download path
mnist_url = 'http://yann.lecun.com/exdb/mnist/'

cwd = os.getcwd()

if os.path.isdir(os.path.join(cwd, 'data')):
    pass
else:
    os.mkdir('data')

mnist_url = 'http://yann.lecun.com/exdb/mnist/'
datapath = cwd + '/data/'



def mnist():
    data_directory = os.getcwd() + '/data/'
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


def plot_image(data, imageNumber):
    plt.imshow(data[:, imageNumber].reshape(28, 28))

def get_classic_3():
    file = 'classic3.csv'
    data = pd.read_csv(file)
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
    matlab_dict = io.loadmat(file)
    data_matrix = np.array(matlab_dict['fea'])
    ground_truth = np.array(matlab_dict['gnd']).reshape(1,-1)[0]
    return data_matrix, ground_truth

def prepare_for_mnist():
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, 'data')):
        pass
    else:
        os.mkdir('data')
    download_parse('train-images-idx3-ubyte.gz')
    download_parse('train-labels-idx1-ubyte.gz')


def download_parse(fgz):
    if os.path.exists(os.path.join(datapath, fgz)):
        pass
    else:
        url = urljoin(mnist_url, fgz)
        filename = os.path.join(datapath, fgz)
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







