# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 04:36:38 2020

@author: jpzxshi
"""
import itertools

import numpy as np
import torch
from sklearn import gaussian_process as gp

from utils import TorchData
from math_ops import cross_entropy, cdist, inverse_modulus_continuity


class IMCData(TorchData):
    '''Data for studying the inverse of the modulus of continuity on [0, 1]^{dim}.
    Points are randomly labeled into 'K' tags according to the RBF kernel with 'length_scale'.
    '''
    def __init__(self, dim=1, K=2, length_scale=1, mode='uniform', train_num=100):
        super(IMCData, self).__init__()
        self.dim = dim
        self.K = K
        self.length_scale = length_scale
        self.mode = mode
        self.train_num = train_num
        
        self.__init_data()
    
    def __init_data(self):
        self.X_train = self.__generate_points()
        self.y_train = self.__tag(self.X_train)
        self.X_test = np.zeros([1, self.dim])
        self.y_test = None
        
    def __generate_points(self):
        if self.dim == 1:
            if self.mode == 'uniform':
                return np.linspace(0, 1, num=self.train_num)[:, None]
            elif self.mode == 'random':
                return np.random.rand(self.train_num)[:, None]
            else: raise ValueError
        elif self.dim == 2:
            N = int(np.floor(np.sqrt(self.train_num)))
            if self.mode == 'uniform':
                return np.array(list(itertools.product(range(N), range(N)))) / (N - 1)
            elif self.mode == 'random':
                return np.random.rand(self.train_num, self.dim)
            else: raise ValueError
        else: raise ValueError
        
    def __tag(self, x):
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + 1e-13 * np.eye(A.shape[0]))
        while True:    
            y = np.argmax(L @ np.random.randn(A.shape[0], self.K), axis=1)
            if np.max(y) != np.min(y): break
        return np.eye(self.K)[y]
    
    @property
    def deltaT(self):
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
        return dmin
    
    @staticmethod
    def compute_imc(data, net, output):
        Lmax = np.max(cross_entropy(data.y_train_np, net(data.X_train).cpu().detach().numpy()))
        if net.device == 'cpu':
            f = lambda x: net(torch.DoubleTensor(x)).detach().numpy()
        else:
            f = lambda x: net(torch.cuda.DoubleTensor(x)).cpu().detach().numpy()
        Nx = 10000 if data.dim == 1 else 1000
        imc = inverse_modulus_continuity(f, data.dim, max(np.exp(-Lmax) - 0.5, 0), Nx, True)
        output.append([Lmax, imc])       
        print('{:<9}max_loss: {:<27}deltaf: {:<25}'.format('', Lmax, imc))
        to_stop = True if len(output) > 1 and output[-1][-1] < output[-2][-1] else False
        return to_stop
    
    @staticmethod
    def restore_imc(imc_his, loss_his):
        best_imc_index = np.argmax(imc_his[:, 1])
        epoch = int(loss_his[best_imc_index, 0])
        max_loss, imc = imc_his[best_imc_index]
        print('Model with max deltaf at epoch {}:'.format(epoch))
        print('max_loss:', max_loss, 'deltaf:', imc)
        net = torch.load('model/model{}.pkl'.format(epoch))
        torch.save(net, 'model_best.pkl')
        return net
        

def main():
    data = IMCData(dim=2, K=3, length_scale=0.2, mode='uniform', train_num=16)
    print('X_train\n', data.X_train)
    print('y_train\n', data.y_train)
    print('y_train(mat)\n', np.argmax(data.y_train, axis=1).reshape([4, 4]))

if __name__ == '__main__':
    main()

