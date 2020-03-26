# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 04:36:38 2020

@author: jpzxshi
"""
import itertools

import numpy as np
import torch

from utils import TorchData
from math_ops import cross_entropy, cdist, inverse_modulus_continuity


class ConsData(TorchData):
    '''Data for studying the consistency between the smoothness and the test loss.
    '''
    def __init__(self, dim=1, delta0=0.1, train_num=20, test_num=10000):
        super(ConsData, self).__init__()
        self.dim = dim
        self.delta0 = delta0
        self.train_num = train_num
        self.test_num = test_num
        
        self.__init_data()
    
    def __init_data(self):
        self.X_train = self.__generate_tr_points()
        self.y_train = self.__tag(self.X_train)
        self.X_test = self.__generate_te_points()
        self.y_test = self.__tag(self.X_test)
        
    def __generate_tr_points(self):
        if self.dim == 1:
            n = self.train_num // 2
            return np.hstack((
                    np.linspace(0, 1/2-self.delta0/2, num=n, endpoint=True),
                    np.linspace(1/2+self.delta0/2, 1, num=n, endpoint=True)
                    ))[:, None]
        elif self.dim == 2:
            n = int(np.floor(np.sqrt(self.train_num)))
            xs = np.linspace(0, 1, num=n)
            res = np.array(list(itertools.product(xs, xs)))
            return res[self.__tag(res)[:, 0] != 0.5]
        else: raise ValueError
        
    def __generate_te_points(self):
        if self.dim == 1:
            return np.linspace(0, 1, num=self.test_num)[:, None]
        elif self.dim == 2:
            n = int(np.floor(np.sqrt(self.test_num)))
            xs = np.linspace(0, 1, num=n)
            return np.array(list(itertools.product(xs, xs)))
        else: raise ValueError
        
    def __tag(self, x):
        if self.dim == 1:
            def tag_one(x):
                return [1, 0] if x[0] <= 0.5 - self.delta0 / 2 else \
                       [0.5, 0.5] if x[0] < 0.5 + self.delta0 / 2 else \
                       [0, 1]
            return np.array(list(map(tag_one, x)))
        elif self.dim == 2:
            def tag_one(x):
                return [1, 0] if np.linalg.norm(x - 0.5) <= 0.4 - self.delta0 / 2 else \
                       [0, 1] if np.linalg.norm(x - 0.5) >= 0.4 + self.delta0 / 2 else \
                       [0.5, 0.5]
            return np.array(list(map(tag_one, x)))
        else: raise ValueError
    
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
        Nx = 10000 if data.dim == 1 else 2000
        imc = inverse_modulus_continuity(f, data.dim, max(np.exp(-Lmax) - 0.5, 0), Nx, True)
        output.append([Lmax, imc])
        print('{:<9}max_loss: {:<27}deltaf: {:<25}'.format('', Lmax, imc))
        to_stop = False
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
    data = ConsData(dim=1, delta0=0.1, train_num=10, test_num=21)
    print('X_train\n', data.X_train)
    print('y_train\n', data.y_train)
    print('X_test\n', data.X_test)
    print('y_test\n', data.y_test)

if __name__ == '__main__':
    main()

