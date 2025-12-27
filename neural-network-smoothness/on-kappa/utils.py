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
    
@timing
def run(data, net, criterion, optimizer, epochs, print_every=1000, save=False, callback=None):
    print('Training...')
    loss_history = []
    for i in range(epochs + 1):
        loss = criterion(data.y_train, net(data.X_train))
        if i % print_every == 0 or i == epochs:
            loss_test = criterion(data.y_test, net(data.X_test))
            loss_history.append([i, loss.item(), loss_test.item()])
            print('{:<9}train_loss: {:<25}test_loss: {:<25}'.format(i, loss.item(), loss_test.item()))
            if save:
                if not os.path.exists('model'): os.mkdir('model')
                torch.save(net, 'model/model{}.pkl'.format(i))
            if torch.any(torch.isnan(loss)): return None
            if callback is not None: 
                to_stop = callback(data, net)
                if to_stop: break
        if i < epochs:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    loss_history = np.array(loss_history)
    np.savetxt('loss.txt', loss_history)
    print('Done!')
    return loss_history

def restore(loss):
    best_loss_index = np.argmin(loss[:, 1])
    epoch = int(loss[best_loss_index, 0])
    loss_train, loss_test = loss[best_loss_index, 1], loss[best_loss_index, 2]
    print('Best model at epoch {}:'.format(epoch))
    print('Train loss:', loss_train, 'Test loss:', loss_test)
    net = torch.load('model/model{}.pkl'.format(epoch), weights_only=False)
    torch.save(net, 'model_best.pkl')
    return net

class TorchData:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.device = None
        
    def set_device(self, d='cpu'):
        if d == 'cpu':
            self.all_to_cpu()
        elif d == 'gpu':
            self.all_to_gpu()
        else:
            raise ValueError
        self.device = d
        
    def all_to_cpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.tensor(getattr(self, d), device=torch.device('cpu')))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cpu())
            else:
                pass
    
    def all_to_gpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.tensor(getattr(self, d), device=torch.device('cuda')))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cuda())
            else:
                pass
            
    def all_to_np(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cpu().detach().numpy())
            else:
                pass
    
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


        
def main():
    pass

if __name__ == '__main__':
    main()