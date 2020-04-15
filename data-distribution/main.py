# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 06:45:32 2020

@author: jpzxshi
"""
import itertools

import numpy as np
import torch

from fnn import FNN
from data import MNIST
from math_ops import Cross_entropy, Test_accuracy
from utils import TorchRunner


def one_running(device, data, hlayers, width, activation, optimizer, lr, iterations, batch_size, print_every):
    data.set_device(device)
    data.to_double()
    
    net = FNN(data.dim, data.K, hlayers, width, activation, True)
    net.set_device(device)
    net.to_double()
    
    criterion = Cross_entropy
    
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else: raise NotImplementedError
    
    def callback(data, net):
        print('{:<9}Test_accuracy:{}'.format('', Test_accuracy(data, net)))
    
    tr = TorchRunner(data, net, criterion, optimizer, iterations, batch_size, print_every, True, callback)
    tr.run()
    net = tr.restore()
    
    accuracy = Test_accuracy(data, net)
    print('Test accuracy:', accuracy)

    return accuracy


def main():
    data = MNIST()
    print('MNIST')
    print('X_train:', data.X_train_np.shape, 'y_train:', data.y_train_np.shape)
    print('X_test:', data.X_test_np.shape, 'y_test:', data.y_test_np.shape)
    print('CC:', data.CC)
    
    device = 'gpu' # 'cpu' or 'gpu'
    # net (fnn)
    hlayers = [2, 3]
    width = [256, 512]
    activation = 'relu'
    # training
    optimizer = 'adam'
    lr = [0.001, 0.0001, 0.00001]
    iterations = 10000
    batch_size = 300
    print_every = 100
    
    accuracy_list = []
    for it in itertools.product(hlayers, width, lr):
        h, w, l = it
        accuracy = one_running(device, data, h, w, activation, optimizer, l, iterations, batch_size, print_every)
        accuracy_list.append(accuracy)
    best_accuracy = np.max(accuracy_list)
    print('\nAll done!')
    print('Dataset: MNIST')
    print('CC:', data.CC)
    print('Best accuracy achieved:', best_accuracy)
    
    np.savetxt('CC-accuracy.txt', [data.CC, best_accuracy])
    
    
if __name__ == '__main__':
    main()