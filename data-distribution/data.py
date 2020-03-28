# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 04:36:38 2020

@author: jpzxshi
"""
import os

import numpy as np

from utils import timing, TorchData
from math_ops import cdist
from data_load import load_MNIST

class CCData(TorchData):
    '''Data for studying the cover complexity.
    '''
    def __init__(self):
        super(CCData, self).__init__()
        
        self.__dists = None
        self.__deltaT = None
        self.__TC = None
        self.__SC = None
        self.__MC = None
        self.__CD = None
        self.__CC = None
        
    @property
    def dim(self):
        return self.X_train_np.shape[1]
    
    @property
    def K(self):
        return self.y_train_np.shape[1]
    
    @property
    def dists(self):
        if self.__dists is None:
            self.__dists = cdist(self.X_test_np, self.X_train_np, metric='euclidean')
        return self.__dists
    
    @property
    def deltaT(self):
        if self.__deltaT is None:
            X_train = self.X_train_np
            y_train = self.y_train_np
            dim = X_train.shape[1]
            K = y_train.shape[1]
            dmin = np.sqrt(dim)
            for i in range(K):
                for j in range(i + 1, K):
                    mask_i = np.argmax(y_train, axis=1) == i
                    mask_j = np.argmax(y_train, axis=1) == j
                    if np.any(mask_i) and np.any(mask_j):
                        dmin = min(dmin, np.min(cdist(X_train[mask_i], X_train[mask_j])))
            self.__deltaT = dmin
        return self.__deltaT
    
    @property
    def TC(self):
        if self.__TC is None:
            self.__TC = CCData.rho(self.dists, self.dim)
        return self.__TC
    
    @property
    def SC(self):
        if self.__SC is None:
            rho_list = []
            for i in range(self.K):
                rho_list.append(CCData.rho(self.__get_label_dists(i, i), self.dim))
            self.__SC = np.mean(rho_list)
        return self.__SC

    @property
    def MC(self):
        if self.__MC is None:
            rho_list = []
            for i in range(self.K):
                for j in range(self.K):
                    if i != j:
                        rho_list.append(CCData.rho(self.__get_label_dists(i, j), self.dim))
            self.__MC = np.mean(rho_list)
        return self.__MC            
    
    @property
    def CD(self):
        if self.__CD is None:
            self.__CD = self.SC - self.MC
        return self.__CD
    
    @property
    def CC(self):
        if self.__CC is None:
            print('Computing CC...')
            @timing
            def Computing():
                return (1 - self.TC) / self.CD
            self.__CC = Computing()
        return self.__CC
    
    
    def __get_label_dists(self, test_label, train_label):
        mask_test = np.argmax(self.y_test_np, axis=1) == test_label
        mask_train = np.argmax(self.y_train_np, axis=1) == train_label
        return self.dists[mask_test, :][:, mask_train]
    
    @staticmethod
    def rho(dists, dim, n=1000):
        '''dists: [test_n, train_n]
        '''
        h = lambda r: np.mean(np.any(dists < r, axis=1))
        diam = np.sqrt(dim)
        step = diam / n
        return np.sum(list(map(h, np.arange(0, diam, step)))) * step / diam
    
    


class MNIST(CCData):
    '''Dataset MNIST.
    '''
    def __init__(self):
        super(MNIST, self).__init__()
        # load data
        path = os.getcwd() + '/datasets/mnist_data/'
        mnist = load_MNIST(path, pixel_normalization=True, one_hot=True)
        
        self.X_train = mnist['X_train'][5000:]    #(55000, 784)
        self.y_train = mnist['y_train'][5000:]    #(55000, 10)
        
        self.X_test = mnist['X_test']             #(10000, 784)
        self.y_test = mnist['y_test']             #(10000, 10)
        
        

def main():
    mnist = MNIST()
    print('MNIST')
    print('X_train:', mnist.X_train_np.shape, 'y_train:', mnist.y_train_np.shape)
    print('X_test:', mnist.X_test_np.shape, 'y_test:', mnist.y_test_np.shape)
    print('CC:', mnist.CC)
    print('TC:', mnist.TC, 'SC:', mnist.SC, 'MC:', mnist.MC, 'CD:', mnist.CD)

if __name__ == '__main__':
    main()

