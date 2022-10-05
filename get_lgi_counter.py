import torch
import os

save_path = './result/non_lgi'
id = str(torch.rand(1).item())

# def func(x):
#     return torch.exp(-1./torch.abs(x))

def func(x):
    return (x**4).pow(1/3)/4 + x**2/2


error = 1
epoch = 0
lr = 1e-2
train_loss_list, norm_list = [], []
x0 = torch.ones(size=(1,))
x0.requires_grad = True
while epoch < 1000:
    loss = func(x0).item()
    error = loss
    grad = torch.autograd.grad(func(x0), x0)[0]
    norm = torch.abs(grad).item()
    print('epoch: {}, training loss: {}, gradient norm: {}'.format(epoch, loss, norm))
    train_loss_list.append(loss)
    norm_list.append(norm)
    x0 = x0 - lr * grad
    epoch += 1


torch.save({
            'train_loss': train_loss_list,
            'gradient_norm': norm_list,
            }, os.path.join(save_path, id))
