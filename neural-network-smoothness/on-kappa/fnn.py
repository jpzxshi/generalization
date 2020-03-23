# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:42:34 2020

@author: jpzxshi
"""
import torch
import torch.nn as nn

from utils import TorchModule


class FNN(TorchModule):
    '''Fully connected neural networks.
    '''
    def __init__(self, ind, outd, hlayers=2, width=50, activation='relu', softmax=False, double=False):
        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.hlayers = hlayers
        self.width = width
        self.activation = activation
        self.softmax = softmax
        
        self.modus = self.__init_modules()
        if double: self.to(torch.double)
        
    def forward(self, x):
        for i in range(self.hlayers):
            LinM = self.modus['LinM{}'.format(i + 1)]
            NonM = self.modus['NonM{}'.format(i + 1)]
            x = NonM(LinM(x))
        x = self.modus['LinMout'](x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=1) if len(x.size()) == 2 else nn.functional.softmax(x, dim=0)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.hlayers > 0:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            modules['NonM1'] = self.Act()
            for i in range(1, self.hlayers):
                modules['LinM{}'.format(i + 1)] = nn.Linear(self.width, self.width)
                modules['NonM{}'.format(i + 1)] = self.Act()
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
        return modules
    
    def Act(self):
        if self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'elu':
            return nn.ELU()
        else:
            raise NotImplementedError
        
    
def main():
    net = FNN(ind=3, outd=3, hlayers=0)
    print(net)
    net = FNN(ind=2, outd=3, hlayers=2, width=10, activation='relu', softmax=True)
    print(net)
    net.set_device('cpu')
    x = torch.randn([5, 2]).cpu()
    print(net(x))

if __name__ == '__main__':
    main()