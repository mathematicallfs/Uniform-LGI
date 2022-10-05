import os
import numpy as np
from scipy import stats
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 600


# get estimated theta and c for the whole data by fitting linear regression
def linear_fit(loss, norm, min_loss):
    x = np.log(loss - min_loss)
    y = np.log(norm)
    z = np.polyfit(x, y, 1)
    theta = z[0]
    pred = np.poly1d(z)(x)
    logc = z[1] + (y - pred).min()

    return theta, np.exp(logc)


# get estimated theta and c by finite sample test
def finite_sample_test(loss, norm, min_loss, K0, s):
    theta_list, c_list, epoch = [], [], []
    for i in np.arange(K0, np.shape(loss)[0]-1, s):
        epoch.append(i)
        theta, c = linear_fit(loss[:i], norm[:i], min_loss)
        theta_list.append(theta)
        c_list.append(c)
    
    # the whole data set
    final_epoch = np.shape(loss)[0]
    epoch.append(final_epoch)
    idx = loss.argmin()
    theta, c = linear_fit(np.delete(loss, idx), np.delete(norm, idx), min_loss)
    theta_list.append(theta)
    c_list.append(c)

    return theta_list, c_list, epoch


def plot(name, K0, s):
    path = os.path.join('./result/', name)
    path = os.path.join(path, 'SGD')
    train_loss = [np.array(torch.load(os.path.join(path, i))['train_loss']) for i in os.listdir(path)]
    if name == 'ResNet18' or name == 'Wide_ResNet':
        min_loss = [l.min() for l in train_loss]
    else:
        min_loss = [0]*len(train_loss)
    gradient_norm = [np.array(torch.load(os.path.join(path, i))['gradient_norm']) for i in os.listdir(path)]

    if name == 'Non_analytic':
        theta, c, epoch = finite_sample_test(train_loss[0], gradient_norm[0], min_loss[0], K0, s)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$K$', size=15)
        ax.set_xscale('log')
        p1 = ax.plot(epoch, theta, label=r'$\theta$', linewidth=3, color='#1f77b4')
        ax2 = ax.twinx()
        p2 = ax2.plot(epoch, c, label=r'$c$', linewidth=3, color='#ff7f0e')
        ps = p1+p2
        labels = [p.get_label() for p in ps]
        ax.legend(ps, labels, loc=0, prop={'size': 15})
        ax.tick_params(axis='y', colors=p1[0].get_color(), labelsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax2.tick_params(axis='y', colors=p2[0].get_color(), labelsize=15)
        ax.set_title(r'{}: $\theta^* = {}, c^* = {}$'.format(r'$e^{-\frac{1}{|w|}}$', theta[-1].round(3), c[-1].round(3)), fontsize=15)
        fig.tight_layout()
        plt.show()

    if name == 'non_lgi':
        theta, c, epoch = finite_sample_test(train_loss[0], gradient_norm[0], min_loss[0], K0, s)
        plt.plot(epoch, theta, label=r'$\theta$', linewidth=3)
        plt.plot(epoch, c, label=r'$c$', linewidth=3)
        plt.legend(prop={'size': 15})
        plt.xlabel(r'$K$', size=15)
        plt.ylabel('estimated value', size=15)
        thetastar = np.array(theta[-6:-1]).mean()
        cstar = np.array(c[-6:-1]).mean()
        plt.title(r'{}: $\theta^* = {}, c^* = {}$'.format(r'$\frac{1}{4}w^{\frac{4}{3}}+\frac{1}{2}w^2$', 
                    thetastar.round(3), cstar.round(3)), fontsize=15)
        plt.tick_params(labelsize=15)
        plt.tight_layout()
        plt.show()

    else:
        theta_list, c_list = [], []
        for id in range(10):
            theta, c, epoch = finite_sample_test(train_loss[id], gradient_norm[id], min_loss[id], K0, s)
            p1 = plt.plot(epoch, theta, label=r'$\theta$', linewidth=3)
            p2 = plt.plot(epoch, c, label=r'$c$', linewidth=3)
            plt.legend(prop={'size': 15})
            plt.xlabel(r'$K$', size=15)
            plt.ylabel('estimated value', size=15)
            thetastar = np.array(theta[-6:-1]).mean()
            cstar = np.array(c[-6:-1]).mean()
            plt.title(r'{}: $\theta^* = {}, c^* = {}$'.format(name, 
                        thetastar.round(3), cstar.round(3)), fontsize=15)
            plt.tick_params(labelsize=15)
            plt.tight_layout()
            plt.show()

            theta_list.append(thetastar)
            c_list.append(cstar)

        theta_lim = np.array(theta_list)
        c_lim = np.array(c_list)
        print(theta_lim.mean(), stats.sem(theta_lim))
        print(c_lim.mean(), stats.sem(c_lim))


# name = 'Non_analytic, non_lgi, Linear_regression, 2_MLP, ResNet18, Wide_ResNet'
plot(name='ResNet18', K0=50, s=1)
