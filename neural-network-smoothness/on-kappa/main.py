# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 06:45:32 2020

@author: jpzxshi
"""
import os
import time
import itertools

import numpy as np
import torch

from data import IMCData
from fnn import FNN
from utils import run
from math_ops import cross_entropy_loss

def one_running(device, dim, K, length_scale, mode, hlayers, width, activation, 
                lr, epochs, print_every, train_num):
    
    data = IMCData(dim=dim, K=K, length_scale=length_scale, mode=mode, train_num=train_num)
    data.set_device(device)
    
    net = FNN(ind=dim, outd=K, hlayers=hlayers, width=width, activation=activation, softmax=True, double=True)
    net.set_device(device)
    
    criterion = cross_entropy_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    imc = []
    callback = lambda x, y: IMCData.compute_imc(x, y, output=imc)
    
    loss = run(data, net, criterion, optimizer, epochs, print_every=print_every, save=True, callback=callback)
    if loss is None: return None
    imc = np.array(imc)
    net = IMCData.restore_imc(imc, loss)
    
    if dim == 2:
        n = int(np.floor(np.sqrt(train_num)))
        print('True labels\n', np.argmax(data.y_train_np, axis=1).reshape([n, n]))
        print('Trained labels\n', np.argmax(net(data.X_train).cpu().detach().numpy(), axis=1).reshape([n, n]))
    else:
        print('True labels\n', np.argmax(data.y_train_np, axis=1))
        print('Trained labels\n', np.argmax(net(data.X_train).cpu().detach().numpy(), axis=1))        
        
    deltaT = data.deltaT
    print('deltaT: ', deltaT)
    
    np.savetxt('imc.txt', imc)
    max_imc = np.max(imc[:, 1])
    print('deltaf: ', max_imc)
    
    kappa = max_imc / deltaT
    print('deltaf/deltaT: ', kappa)
    
    return kappa



def main():
    device = ['cpu']
    # problem
    dim = [1] # 1 or 2
    K = [2]
    length_scale = [0.2]
    mode = ['uniform'] # 'uniform' or 'random'
    # network
    hlayers = [2]
    width = [200]
    activation = ['relu']
    # training
    lr = [0.001]
    epochs = [10000]
    print_every = [10]
    train_num = [5, 10, 20, 50, 100, 200, 500, 1000] # square number when dim=2 & mode='uniform'
    
    repeat = 50
    
    kappa_table = []
    for i in range(repeat):
        kappa_list = []
        for it in itertools.product(device, dim, K, length_scale, mode, hlayers, width, 
                                    activation, lr, epochs, print_every, train_num):
            while True:
                kappa = one_running(*it)
                if kappa is not None and kappa > 0:
                    kappa_list.append(kappa)
                    break
                print('Encountering Nan or dying, rerunning...')
        kappa_table.append(kappa_list)
    
    info = [("dim", dim),
            ("K", K),
            ("length_scale", length_scale), 
            ("mode", mode),
            ("hlayers", hlayers),
            ("width", width),
            ("activation", activation),
            ("lr", lr),
            ("epochs", epochs), 
            ("print_every", print_every),
            ("train_num", train_num),
            ("repeat", repeat)]
    
    time_now = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    path = './outputs/' + time_now + '/'
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path + 'info.txt', 'w') as f:
        for item in info:
            f.write('{}: {}\n'.format(item[0], str(item[1])))
    np.savetxt(path + 'kappa.txt', np.array(kappa_table))
    
if __name__ == '__main__':
    main()