# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:27:26 2020

@author: jpzxshi
"""
import struct
import gzip

import numpy as np

def load_MNIST(path, pixel_normalization=False, one_hot=False):
    PATH = {}
    PATH['X_train'] = path + '/train-images-idx3-ubyte.gz'
    PATH['y_train'] = path + '/train-labels-idx1-ubyte.gz'
    PATH['X_test'] = path + '/t10k-images-idx3-ubyte.gz'
    PATH['y_test'] = path + '/t10k-labels-idx1-ubyte.gz'
    
    data = {}
    for t in ['X_train', 'X_test']:
        with gzip.open(PATH[t], 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII',f.read(16))
            data[t] = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(num, rows * cols)
    for t in ['y_train', 'y_test']:
        with gzip.open(PATH[t], 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            data[t] = np.frombuffer(f.read(), dtype=np.uint8).copy().astype(np.integer)
            
    if pixel_normalization:
        data['X_train'] = data['X_train'] / 255
        data['X_test'] = data['X_test'] / 255
    if one_hot:
        data['y_train'] = (np.arange(10) == data['y_train'][:, None]).astype(np.integer)
        data['y_test'] = (np.arange(10) == data['y_test'][:, None]).astype(np.integer)
    return data



def main():
    data = load_MNIST('datasets/mnist_data', pixel_normalization=True, one_hot=True)
    print('X_train:', data['X_train'].shape, 'y_train:', data['y_train'].shape)
    print('X_test:', data['X_test'].shape, 'y_test:', data['y_test'].shape)
    print('max:', np.max(data['X_train']), 'min:', np.min(data['X_train']))
    print('y_train[0]:', data['y_train'][0])

if __name__ == '__main__':
    main()