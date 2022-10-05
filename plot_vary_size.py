import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 600



def linear_fit(loss, norm, min_loss):
    x = np.log(loss - min_loss)
    y = np.log(norm)
    z = np.polyfit(x, y, 1)
    theta = z[0]
    pred = np.poly1d(z)(x)
    logc = z[1] + (y - pred).min()

    return theta, np.exp(logc)



def finite_sample_test(loss, norm, min_loss, K0, s):
    theta_list, c_list = [], []
    idx = loss.argmin()
    norm = np.delete(norm, idx)
    loss = np.delete(loss, idx)
    for i in np.arange(K0, np.shape(loss)[0]-1, s):
        theta, c = linear_fit(loss[:i], norm[:i], min_loss)
        theta_list.append(theta)
        c_list.append(c)
    
    # whole data set
    idx = loss.argmin()
    theta, c = linear_fit(loss, norm, min_loss)
    theta_list.append(theta)
    c_list.append(c)

    return theta_list, c_list



def get_theta_c_bound(seed, path, K0, s):
    path = os.path.join(path, str(seed))
    theta_list, c_list = [], []
    bound_list = []
    for size in np.arange(500, 5001, 500):
        train_loss = np.array(torch.load(os.path.join(path, 'n={}'.format(size)))['train_loss'])
        min_loss = train_loss.min()
        gradient_norm = np.array(torch.load(os.path.join(path, 'n={}'.format(size)))['gradient_norm'])
        theta, c = finite_sample_test(train_loss, gradient_norm, min_loss, K0, s)
        theta_list.append(theta[-1])
        c_list.append(c[-1])
        bound = 1/(c[-1]**2 * (1-theta[-1])**2 * np.sqrt(2*size)) + np.sqrt(512/(2*size))
        bound_list.append(bound)  

    return theta_list, c_list, bound_list  




path = './result/2_MLP/vary_size/CIFAR10/final_result'
n = np.arange(500, 5001, 500)
K0 = 1000
s = 100
theta, c, bound = [], [], []
for i in range(5):
    theta_list, c_list, bound_list = get_theta_c_bound(i, path, K0, s)
    theta.append(theta_list)
    c.append(c_list)
    bound.append(bound_list)
theta = np.concatenate(theta).reshape((5, 10))
c = np.concatenate(c).reshape((5, 10))
bound = np.concatenate(bound).reshape((5, 10))
theta_mean, theta_std = theta.mean(0), theta.std(0)
c_mean, c_std = c.mean(0), c.std(0)
bound_mean, bound_std = bound.mean(0), bound.std(0)



plt.plot(2*n, theta_mean, label=r'$\theta_{n,\delta}$')
plt.plot(2*n, c_mean, label=r'$c_{n,\delta}$')
plt.fill_between(2*n, theta_mean-theta_std, theta_mean+theta_std, alpha=0.5)
plt.fill_between(2*n, c_mean-c_std, c_mean+c_std, alpha=0.5)
plt.legend(prop={'size': 15})
plt.xlabel(r'Sample size $n$', size=15)
plt.ylabel('Uniform-LGI constants', size=15)
plt.ylim(0, 1)
plt.title('CIFAR10 (airplane vs automobile)', fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()

plt.plot(2*n, bound_mean, label=r'$\frac{1}{\sqrt{n} c_{n, \delta}^2 (1 - \theta_{n, \delta})^2} + \sqrt{\frac{m}{n}}$')
plt.fill_between(2*n, bound_mean-bound_std, bound_mean+bound_std, alpha=0.5)
plt.legend(prop={'size': 15})
plt.xlabel(r'Sample size $n$', size=15)
plt.ylabel('Generalization bound', size=15)
plt.title('CIFAR10 (airplane vs automobile)', fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()
