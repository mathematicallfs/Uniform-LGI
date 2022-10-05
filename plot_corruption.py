import os
import numpy as np
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


def finite_sample_test(loss, norm, min_loss, K0, s):
    theta_list, c_list, epoch = [], [], []
    idx = loss.argmin()
    norm = np.delete(norm, idx)
    loss = np.delete(loss, idx)
    for i in np.arange(K0, np.shape(loss)[0]-1, s):
        theta, c = linear_fit(loss[:i], norm[:i], min_loss)
        theta_list.append(theta)
        c_list.append(c)
    
    # the whole data set
    idx = loss.argmin()
    theta, c = linear_fit(loss, norm, min_loss)
    theta_list.append(theta)
    c_list.append(c)

    return theta_list, c_list




path = './result/2_MLP/Corruption/CIFAR10/final_result'
ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
K0 = 1000
s = 100
def get_theta_c(path, seed):
    path = os.path.join(path, str(seed))
    train_loss = [np.array(torch.load(os.path.join(path, 'r={}'.format(r)))['train_loss']) for r in ratio]
    min_loss = [l.min() for l in train_loss]
    gradient_norm = [np.array(torch.load(os.path.join(path, 'r={}'.format(r)))['gradient_norm']) for r in ratio]
    theta_list, c_list = [], []
    r = np.arange(6)*0.1
    for id in range(6):
        theta, c = finite_sample_test(train_loss[id], gradient_norm[id], min_loss[id], K0, s)
        theta_list.append(theta[-1])
        c_list.append(c[-1])
    return theta_list, c_list



theta, c = np.zeros((5,6)), np.zeros((5,6))
for i in range(5):
    theta[i,:] = get_theta_c(path, i)[0]
    c[i,:] = get_theta_c(path, i)[1]

theta_mean = np.array([theta[:,i].mean() for i in range(6)])
theta_std = np.array([theta[:,i].std() for i in range(6)])
c_mean = np.array([c[:,i].mean() for i in range(6)])
c_std = np.array([c[:,i].std() for i in range(6)])

plt.plot(ratio, theta_mean, label=r'$\theta_{n,\delta}$')
plt.plot(ratio, c_mean, label=r'$c_{n,\delta}$')
plt.fill_between(ratio, theta_mean-theta_std, theta_mean+theta_std, alpha=0.5)
plt.fill_between(ratio, c_mean-c_std, c_mean+c_std, alpha=0.5)
plt.legend(prop={'size': 15})
plt.xlabel('Ratio of label flip', size=15)
plt.ylabel('Uniform-LGI constants', size=15)
plt.title('CIFAR10 (airplane vs automobile)', fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()

