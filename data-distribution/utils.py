# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 02:37:43 2019

@author: jpzxshi
"""
import os
from functools import wraps
import time

import numpy as np
import torch

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(func.__name__ + ' took {} s'.format(time.time() - t))
        return result
    return wrapper

class TorchRunner:
    def __init__(self, data, net, criterion, optimizer, iterations, 
                 batch_size=None, print_every=1000, save=False, callback=None):
        self.data = data
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.iterations = iterations
        self.batch_size = batch_size
        self.print_every = print_every
        self.save = save
        self.callback = callback
        
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
    
    @timing
    def run(self):
        print('Training...')
        loss_history = []
        for i in range(self.iterations + 1):
            if self.batch_size is not None:
                mask = np.random.choice(self.data.X_train.size(0), self.batch_size, replace=False)
                loss = self.criterion(self.data.y_train[mask], self.net(self.data.X_train[mask]))
            else:
                loss = self.criterion(self.data.y_train, self.net(self.data.X_train))
            if i % self.print_every == 0 or i == self.iterations:
                loss_test = self.criterion(self.data.y_test, self.net(self.data.X_test))
                loss_history.append([i, loss.item(), loss_test.item()])
                print('{:<9}Train_loss: {:<25}Test_loss: {:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None                
                if self.save:
                    if not os.path.exists('model'): os.mkdir('model')
                    torch.save(self.net, 'model/model{}.pkl'.format(i))
                if self.callback is not None: 
                    to_stop = self.callback(self.data, self.net)
                    if to_stop: break
            if i < self.iterations:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.loss_history = np.array(loss_history)
        np.savetxt('loss.txt', self.loss_history)
        print('Done!')
        return self.loss_history
    
    def restore(self):
        if self.loss_history is not None and self.save == True:
            best_loss_index = np.argmin(self.loss_history[:, 1])
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]
            print('Best model at iteration {}:'.format(iteration))
            print('Train loss:', loss_train, 'Test loss:', loss_test)
            self.best_model = torch.load('model/model{}.pkl'.format(iteration))
            torch.save(self.best_model, 'model_best.pkl')
        else:
            raise RuntimeError('restore before running or without saved models')
        return self.best_model

class TorchData:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.device = None
        
    def set_device(self, d):
        if d == 'cpu':
            self.__to_cpu()
        elif d == 'gpu':
            self.__to_gpu()
        else:
            raise ValueError
        self.device = d
        
    def to_float(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).float())
        
    def to_double(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).double())

    @property
    def dim(self):
        if isinstance(self.X_train, np.ndarray):
            return self.X_train.shape[-1]
        elif isinstance(self.X_train, torch.Tensor):
            return self.X_train.size(-1)
    
    @property
    def K(self):
        if isinstance(self.y_train, np.ndarray):
            return self.y_train.shape[-1]
        elif isinstance(self.y_train, torch.Tensor):
            return self.y_train.size(-1)
    
    @property
    def X_train_np(self):
        return TorchData.to_np(self.X_train)
            
    @property
    def y_train_np(self):
        return TorchData.to_np(self.y_train)
            
    @property
    def X_test_np(self):
        return TorchData.to_np(self.X_test)
            
    @property
    def y_test_np(self):
        return TorchData.to_np(self.y_test)
    
    @staticmethod      
    def to_np(d):
        if isinstance(d, np.ndarray) or d is None:
            return d
        elif isinstance(d, torch.Tensor):
            return d.cpu().detach().numpy()
        else:
            raise ValueError
            
    def __to_cpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.FloatTensor(getattr(self, d)))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cpu())
    
    def __to_gpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.cuda.FloatTensor(getattr(self, d)))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cuda())

class TorchModule(torch.nn.Module):
    def __init__(self):
        super(TorchModule, self).__init__()
        self.device = None
    
    def set_device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self.cuda()
        else:
            raise ValueError
        self.device = d
        
    def to_float(self):
        self.to(torch.float)
        
    def to_double(self):
        self.to(torch.double)


        
def main():
    pass

if __name__ == '__main__':
    main()