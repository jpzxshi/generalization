# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 06:45:32 2020

@author: jpzxshi
"""

from data import MNIST



def main():
    mnist = MNIST()
    print('MNIST')
    print('X_train:', mnist.X_train_np.shape, 'y_train:', mnist.y_train_np.shape)
    print('X_test:', mnist.X_test_np.shape, 'y_test:', mnist.y_test_np.shape)
    print('CC:', mnist.CC)
    
if __name__ == '__main__':
    main()