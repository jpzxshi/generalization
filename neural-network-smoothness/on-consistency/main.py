# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 06:45:32 2020

@author: jpzxshi
"""
import os
import time

import numpy as np
import torch

from data import ConsData
from fnn import FNN
from utils import run
from math_ops import cross_entropy_loss
from plot import plot_figure

def one_running(device, dim, K, delta0, hlayers, width, activation, lr, epochs, 
                print_every, train_num, test_num):
    
    data = ConsData(dim=dim, delta0=delta0, train_num=train_num, test_num=test_num)
    data.set_device(device)
    
    net = FNN(ind=dim, outd=K, hlayers=hlayers, width=width, activation=activation, softmax=True, double=True)
    net.set_device(device)
    
    criterion = cross_entropy_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    imc = []
    callback = lambda x, y: ConsData.compute_imc(x, y, output=imc)
    
    loss = run(data, net, criterion, optimizer, epochs, print_every=print_every, save=True, callback=callback)
    if loss is None: return None
    
    deltaT = data.deltaT
    print('deltaT: ', deltaT)
    
    deltaf = np.array(imc)
    max_deltaf = np.max(deltaf[:, 1])
    print('Max deltaf: ', max_deltaf)
    
    plot_figure(loss, deltaf, deltaT)
    
    return [loss, deltaf, deltaT]



def main():
    device = 'cpu'
    # problem
    dim = 1 # 1 or 2
    K = 2
    delta0 = 0.1
    # network
    hlayers = 2
    width = 30
    activation = 'relu'
    # training
    lr = 0.001
    epochs = 1000 # 1000 for dim=1 and 2000 for dim=2
    print_every = 10
    train_num = 20 # 20 for dim=1 and 400 for dim=2
    test_num = 10000 # 10000 for dim=1 and 1000000 for dim=2
    
    params = [device, dim, K, delta0, hlayers, width, activation, lr, epochs, 
             print_every, train_num, test_num]
    while True:
        res = one_running(*params)
        if res is not None: break
        print('Encountering Nan, rerunning...')
    
    info = [("dim", dim),
            ("K", K),
            ("delta0", delta0),
            ("hlayers", hlayers),
            ("width", width),
            ("activation", activation),
            ("lr", lr),
            ("epochs", epochs), 
            ("print_every", print_every),
            ("train_num", train_num),
            ("test_num", test_num)]
    
    time_now = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    path = './outputs/' + time_now + '/'
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path + 'info.txt', 'w') as f:
        for item in info:
            f.write('{}: {}\n'.format(item[0], str(item[1])))
    np.savetxt(path + 'loss.txt', np.array(res[0]))
    np.savetxt(path + 'deltaf.txt', np.array(res[1]))
    np.savetxt(path + 'deltaT.txt', np.array([res[2]]))
    
if __name__ == '__main__':
    main()