# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 02:37:43 2019

@author: jpzxshi
"""
import os
import itertools

import numpy as np
import torch


#
# numpy
#
def cross_entropy(p, q):
    return -np.sum(p * np.log(q), axis=1)


def softmax(X):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(X - np.max(X, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]


def inverse_modulus_continuity(f, dim, eps, Nx=1000, save_iters=False):
    """Compute the inverse of the modulus of continuity.
    The domain of 'f' is [0, 1]^{dim}.
    """
    if dim == 1:
        X = np.linspace(0, 1, num=Nx)[:, None]
        Y = f(X)
        dx = X[1, 0] - X[0, 0]
        k = 0
        while k < Nx:
            if np.any(np.linalg.norm(Y[:Nx-k] - Y[k:], ord=np.inf, axis=1) >= eps):
                return k * dx
            k += 1
        return 1
    elif dim == 2:
        iters = list(itertools.product(range(Nx), range(Nx)))
        X = np.array(iters) / (Nx - 1)
        Y = f(X)
        data_iters = 'iters_{}.npy'.format(Y.shape[0])
        if save_iters and os.path.isfile(data_iters):
            iters = np.load(data_iters)
        else:
            iters.sort(key=lambda point: np.linalg.norm(point))
            if save_iters:  
                np.save(data_iters, iters)
                print('save', data_iters)
        Ymat = Y.reshape([Nx, Nx, Y.shape[-1]])
        def runover(m, n):
            dY = np.linalg.norm(Ymat[:Nx - m, :Nx - n] - Ymat[m:, n:], ord=np.inf, axis=2)
            if np.any(dY >= eps):
                return np.linalg.norm([m, n]) / (Nx - 1)
            return None
        for it in iters:
            delta = runover(*it)
            if delta is not None: return delta
        return np.sqrt(dim)
    else: return None
    

def cdist(x, y, metric='euclidean'):
    dists = None
    if metric == 'euclidean':
        num_1, num_2 = x.shape[0], y.shape[0]
        dist_1 = np.sum(np.square(x), axis=1, keepdims=True) * np.ones(num_2)
        dist_2 = np.ones([num_1, 1]) * np.sum(np.square(y), axis=1)
        dist_3 = - 2 * np.dot(x, y.transpose())
        dists = np.sqrt(np.abs(dist_1 + dist_2 + dist_3))
    else: raise NotImplementedError
    return dists

#
# torch
#    
def cross_entropy_loss(p, q):
    if p is not None and q is not None:
        return torch.mean(-torch.sum(p * torch.log(q), dim=1))
    else:
        return torch.DoubleTensor([-1])
         

def grad(y, x):
    '''
    y: [N, Ny] or [Ny]
    x: [N, Nx] or [Nx]
    Return dy/dx ([N, Ny, Nx] or [Ny, Nx]).
    '''
    N = y.size(0) if len(y.size()) == 2 else 1
    Ny = y.size()[-1]
    Nx = x.size()[-1]
    z = torch.ones([N])
    if x.is_cuda:
        z=z.cuda()
    dy = []
    if len(y.size()) == 2:
        for i in range(Ny):
            dy.append(torch.autograd.grad(y[:, i], x, grad_outputs=z, create_graph=True)[0])
        res = torch.cat(dy, 1).view([N, Ny, Nx])
    else:
        for i in range(Ny):
            dy.append(torch.autograd.grad(y[i], x, grad_outputs=z, create_graph=True)[0])
        res = torch.cat(dy, 0).view([Ny, Nx])
    return res

     

   
def main():
    #a = torch.tensor([[1.0, 2], [4, 5]], requires_grad=True)
    #A = torch.tensor([[1.0, -7, 2], [3, 4, 5]])
    #b = torch.nn.functional.relu(a @ A)
    #grads = grad(b, a)
    #print(grads)
    
    f = lambda x: x @ np.array([[1], [1]])
    imc = inverse_modulus_continuity(f, dim=2, eps=0.5, Nx=200, save_iters=False)
    print(imc)
    

if __name__ == '__main__':
    main()
        