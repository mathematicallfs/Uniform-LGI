'''
get the dependence of c_{n, delta} and theta_{n, delta} on the sample size n
also use CIFAR10 with the first two classes on a two-layer MLP model
then the total size for each class is 5000, we consdier varying n = [500, 1000, ..., 5000] (10 groups)
'''

import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.nn as nn
from train_test import train
from networks import mlp


init_path = './result/2_MLP/vary_size/CIFAR10/init_model'
save_path = './result/2_MLP/vary_size/CIFAR10/final_result'
'''
save 5 initial models with different initializations
'''
# seed = 0
# torch.save({'model': model.state_dict()}, os.path.join(init_path, str(seed)))


'''
get data with size n = [500, 1000, ..., 5000]
'''


def get_dataloader(train_dataset, size):
    # select the images with label 0 or 1
    train_idx = []
    for i in [0, 1]:
        idx = np.arange(len(train_dataset))[np.array(train_dataset.targets) == i]
        np.random.shuffle(idx)
        train_idx.append(idx[:size])
    train_idx = np.concatenate(train_idx)
    train_dataset.data = train_dataset.data[train_idx]
    train_dataset.targets = np.array(train_dataset.targets)[train_idx]
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset.transform = transform
    train_loader = DataLoader(train_dataset, batch_size=len(train_idx), shuffle=False)

    return train_loader




def get_final_result(model, train_dataset, size, seed, save_path):
    model.load_state_dict(torch.load(os.path.join(init_path, str(seed)))['model'])
    save_path = os.path.join(save_path, str(seed))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loader = get_dataloader(train_dataset, size)
    criterion = nn.CrossEntropyLoss()
    error = 1
    epoch = 0
    train_loss_list, train_acc_list, norm_list = [], [], []
    while error > 1e-3 and epoch < 20000:
        train_loss, train_acc, norm = train(model, train_loader, criterion, optimizer)
        error = train_loss
        epoch += 1
        print('epoch: {}, training loss: {}, acc: {}, gradient norm: {}'.format(epoch, train_loss, train_acc, norm))
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        norm_list.append(norm)
    torch.save({'train_loss': train_loss_list,
                'train_acc': train_acc_list,
                'gradient_norm': norm_list
                }, os.path.join(save_path, 'n={}'.format(size)))





n = np.arange(500, 5001, 500)
seed = 4
size = n[4]
model = mlp(layer=2, use_bn=False, bias=False, act='ReLU', dims=[3*32*32, 512, 2]).cuda()
train_dataset = datasets.CIFAR10('./data', train=True, download=False)
get_final_result(model, train_dataset, size, seed, save_path)
