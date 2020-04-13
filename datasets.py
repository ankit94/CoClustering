'''
This code os refferred from a course we had taken last semester
'''
import os
import numpy as np
from urllib.request import urlretrieve
from urllib.parse import urljoin
import random
import pandas as pd
from matplotlib import pyplot as plt
random.seed(1)
np.random.seed(1)

# Dataset download path
mnist_url = 'http://yann.lecun.com/exdb/mnist/'

cwd = os.getcwd()

if os.path.isdir(os.path.join(cwd, 'data')):
    pass
else:
    os.mkdir('data')

mnist_url = 'http://yann.lecun.com/exdb/mnist/'
datapath = cwd + '/data/'



def mnist(noSamples=1000, digits=[3, 8]):
    data_dir = os.getcwd() + '/data/'
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    trX = np.zeros((noSamples, 28 * 28))
    trY = np.zeros(noSamples)
    # Normalize the data
    trData = trData / 255.

    if noSamples % len(digits) != 0:
        raise ValueError(
            "Unequal number of samples per class will be returned, adjust noTrSamples and digits accordingly!")
    else:
        noTrPerClass = noSamples // len(digits)

    count = 0
    for ll in range(len(digits)):
        idl = np.where(trLabels == digits[ll])[0]
        np.random.shuffle(idl)
        idl_ = idl[: noTrPerClass]
        idx = list(range(count * noTrPerClass, (count + 1) * noTrPerClass))
        trX[idx, :] = trData[idl_, :]
        trY[idx] = trLabels[idl_]

    train_idx = np.random.permutation(trX.shape[0])
    trX = trX[train_idx, :]
    trY = trY[train_idx]

    trX = trX.T
    trY = trY.reshape(1, -1)

    plot_image(trX, 0)
    return trX, trY

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
    2 : get_mnist
}

def prepare_dataset(dataset_number, **options):
    if not dataset_number:
        return []
    return get_data_set(dataset_number, **options)

def get_data_set(number, **options):
    return dataset[number](**options)







