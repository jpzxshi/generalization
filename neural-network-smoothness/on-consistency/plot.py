# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:16:38 2020

@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_figure(loss, deltaf, deltaT):
    index_min, index_max = np.argmin(loss[:, 2]), np.argmax(deltaf[:, 1])
    xmin_tl, ymin_tl = loss[index_min, 0], loss[index_min, 2]
    xmax_df, ymax_df = loss[index_max, 0], deltaf[index_max, 1] * 10

    plt.figure()
    
    plt.plot(loss[:, 0], loss[:, 2], label='Test loss', color='royalblue')
    plt.plot(loss[:, 0], 10 * deltaf[:, 1], label=r'$10\delta_{f}$', color='red')
    plt.plot(loss[:, 0], 10 * (deltaT / 2) * np.ones_like(loss[:, 0]), '--r', 
             label=r'$10\delta_{\mathcal{T}}/2$')
    plt.annotate('', xy=(xmin_tl, ymin_tl + 0.01), xytext=(xmin_tl, ymin_tl + 0.148),
                 arrowprops=dict(color='royalblue', width=2, headwidth=8, headlength=10))
    plt.annotate('', xy=(xmax_df, ymax_df - 0.01), xytext=(xmax_df, ymax_df - 0.148),
                 arrowprops=dict(color='red', width=2, headwidth=8, headlength=10))
    plt.ylim((0, 0.8))
    plt.xlabel('Iterations', fontdict={'fontsize': 14})
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)

    plt.savefig('figure.pdf', dpi=500)
    
    
    
    
def main():
    # 1d
    loss_one = np.loadtxt('outputs/one_dim/loss.txt')
    deltaf_one = np.loadtxt('outputs/one_dim/deltaf.txt')
    deltaT_one = np.loadtxt('outputs/one_dim/deltaT.txt')
    index_min, index_max = np.argmin(loss_one[:, 2]), np.argmax(deltaf_one[:, 1])
    xmin_tl_one, ymin_tl_one = loss_one[index_min, 0], loss_one[index_min, 2]
    xmax_df_one, ymax_df_one = loss_one[index_max, 0], deltaf_one[index_max, 1] * 10
    
    # 2d
    loss_two = np.loadtxt('outputs/two_dim/loss.txt')
    deltaf_two = np.loadtxt('outputs/two_dim/deltaf.txt')
    deltaT_two = np.loadtxt('outputs/two_dim/deltaT.txt')
    index_min, index_max = np.argmin(loss_two[:, 2]), np.argmax(deltaf_two[:, 1])
    xmin_tl_two, ymin_tl_two = loss_two[index_min, 0], loss_two[index_min, 2]
    xmax_df_two, ymax_df_two = loss_two[index_max, 0], deltaf_two[index_max, 1] * 10
    
    plt.figure(figsize=[6.4 + 0.5, 4.8 * 2])
    # 1d
    plt.subplot(211)
    plt.plot(loss_one[:, 0], loss_one[:, 2], label='Test loss', color='royalblue')
    plt.plot(loss_one[:, 0], 10 * deltaf_one[:, 1], label=r'$10\delta_{f}$', color='red')
    plt.plot(loss_one[:, 0], 10 * (deltaT_one / 2) * np.ones_like(deltaf_one[:, 0]), '--r', 
             label=r'$10\delta_{\mathcal{T}}/2$')
    plt.annotate('', xy=(xmin_tl_one, ymin_tl_one + 0.01), xytext=(xmin_tl_one, ymin_tl_one + 0.148),
                 arrowprops=dict(color='royalblue', width=2, headwidth=8, headlength=10))
    plt.annotate('', xy=(xmax_df_one, ymax_df_one - 0.01), xytext=(xmax_df_one, ymax_df_one - 0.148),
                 arrowprops=dict(color='red', width=2, headwidth=8, headlength=10))
    plt.ylim((0, 0.8))
    plt.xlabel('Iterations', fontdict={'fontsize': 14})
    plt.tick_params(labelsize=14)
    plt.title('(A)', fontdict={'fontsize': 28}, loc='left')
    plt.legend(fontsize=14)
    # 2d
    plt.subplot(212)
    plt.plot(loss_two[:, 0], loss_two[:, 2], label='Test loss', color='royalblue')
    plt.plot(loss_two[:, 0], 10 * deltaf_two[:, 1], label=r'$10\delta_{f}$', color='red')
    plt.plot(loss_two[:, 0], 10 * (deltaT_two / 2) * np.ones_like(deltaf_two[:, 0]), '--r', 
             label=r'$10\delta_{\mathcal{T}}/2$')
    plt.annotate('', xy=(xmin_tl_two, ymin_tl_two + 0.008), xytext=(xmin_tl_two, ymin_tl_two + 0.1108),
                 arrowprops=dict(color='royalblue', width=2, headwidth=8, headlength=10))
    plt.annotate('', xy=(xmax_df_two, ymax_df_two - 0.008), xytext=(xmax_df_two, ymax_df_two - 0.1108),
                 arrowprops=dict(color='red', width=2, headwidth=8, headlength=10))
    plt.ylim((0, 0.7))
    plt.xlabel('Iterations', fontdict={'fontsize': 14})
    plt.tick_params(labelsize=14)
    plt.title('(B)', fontdict={'fontsize': 28}, loc='left')
    plt.legend(fontsize=14)
    #
    plt.tight_layout()
    plt.savefig('one_two_dim.pdf', dpi=500)

if __name__ == '__main__':
    main()