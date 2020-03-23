# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:16:38 2020

@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt
    
    
def depth():
    kappa = np.loadtxt('outputs/depth_w50/kappa.txt')
    x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Width=50', color='limegreen', marker='s', linewidth=3)
    
    kappa = np.loadtxt('outputs/depth_w100/kappa.txt')
    x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Width=100', color='royalblue', marker='s', linewidth=3)
    
    kappa = np.loadtxt('outputs/depth_w200/kappa.txt')
    x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Width=200', color='orangered', marker='s', linewidth=3)
    
    plt.ylim([0, 0.5])
    plt.xlabel(r'Depth', fontdict={'fontsize': 14})
    plt.ylabel(r'$\delta_{f}/\delta_{\mathcal{T}}$', fontdict={'fontsize': 14})
    plt.title('(A)', fontdict={'fontsize': 28}, loc='left')
    plt.legend(fontsize=14, loc=3)


def width():
    kappa = np.loadtxt('outputs/width_h2/kappa.txt')
    x = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Depth=3', color='limegreen', marker='s', linewidth=3)
    
    kappa = np.loadtxt('outputs/width_h3/kappa.txt')
    x = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Depth=4', color='royalblue', marker='s', linewidth=3)
    
    kappa = np.loadtxt('outputs/width_h4/kappa.txt')
    x = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Depth=5', color='orangered', marker='s', linewidth=3)
    
    plt.semilogx()
    plt.ylim([0, 0.5])
    plt.xlabel(r'Width', fontdict={'fontsize': 14})
    plt.ylabel(r'$\delta_{f}/\delta_{\mathcal{T}}$', fontdict={'fontsize': 14})
    plt.title('(B)', fontdict={'fontsize': 28}, loc='left')
    plt.legend(fontsize=14, loc=3)
    
def tr_num():
    kappa = np.loadtxt('outputs/tr_data_h2_w200/kappa.txt')
    x = [5, 10, 20, 50, 100, 200, 500, 1000]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Depth=3 Width=200', color='limegreen', marker='s', linewidth=3)
    
    kappa = np.loadtxt('outputs/tr_data_h4_w200/kappa.txt')
    x = [5, 10, 20, 50, 100, 200, 500, 1000]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Depth=5 Width=200', color='royalblue', marker='s', linewidth=3)

    kappa = np.loadtxt('outputs/tr_data_h4_w500/kappa.txt')
    x = [5, 10, 20, 50, 100, 200, 500, 1000]
    y = np.mean(kappa, axis=0)
    plt.plot(x, y, label=r'Depth=5 Width=500', color='orangered', marker='s', linewidth=3)
    
    plt.semilogx()
    plt.ylim([0, 0.5])
    plt.xlabel(r'Training data', fontdict={'fontsize': 14})
    plt.ylabel(r'$\delta_{f}/\delta_{\mathcal{T}}$', fontdict={'fontsize': 14})
    plt.legend(fontsize=14, loc=3)    
    
    
def main():
    plt.figure(figsize=[6.9, 4.8 * 2])
    plt.subplot(211)
    depth()
    plt.subplot(212)
    width() 
    plt.tight_layout()
    plt.savefig('depth_width.pdf', dpi=500)
    
    plt.figure(figsize=[6.9, 4.5])
    tr_num()
    plt.tight_layout()
    plt.savefig('tr_data.pdf', dpi=500)

if __name__ == '__main__':
    main()