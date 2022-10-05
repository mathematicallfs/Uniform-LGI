import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.nn as nn
from train_test import train, test
from networks import mlp


model = mlp(layer=2, use_bn=False, bias=False, act='ReLU', dims=[3*32*32, 512, 2]).cuda()
init_path = './result/2_MLP/Corruption/CIFAR10/init_model'
save_path = './result/2_MLP/Corruption/CIFAR10/final_result'
'''
save 5 initial models with different initializations
'''
# seed = 0
# torch.save({'model': model.state_dict()}, os.path.join(init_path, str(seed)))


'''
begin to train with indepdent runs
'''
train_dataset = datasets.CIFAR10('./data', train=True, download=True)
test_dataset= datasets.CIFAR10('./data', train=False)
# select the images with label 0 or 1
train_idx = (np.array(train_dataset.targets)==0) | (np.array(train_dataset.targets)==1) 
test_idx = (np.array(test_dataset.targets)==0) | (np.array(test_dataset.targets)==1) 
train_dataset.data = train_dataset.data[train_idx]
train_dataset.targets = np.array(train_dataset.targets)[train_idx]
test_dataset.data = test_dataset.data[test_idx]
test_dataset.targets = np.array(test_dataset.targets)[test_idx]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# label flip with ratio r in training dataset targets
def label_flip(target, r):
    num = target.shape[0]
    index = np.random.choice(num, int(r*num), replace=False) 
    flip = target[index]
    mask = (flip==0)
    flip[mask] = 1
    flip[~mask] = 0
    target[index] = flip
    
    return target


def get_final_result(model, train_dataset, test_dataset, ratio, seed, save_path):
    model.load_state_dict(torch.load(os.path.join(init_path, str(seed)))['model'])
    save_path = os.path.join(save_path, str(seed))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_dataset.targets = label_flip(train_dataset.targets, r=ratio)
    train_dataset.transform, test_dataset.transform = transform, transform
    train_loader = DataLoader(train_dataset, batch_size=len(train_idx), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
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
    test_loss, test_acc = test(model, test_loader, criterion)
    torch.save({'train_loss': train_loss_list,
                'train_acc': train_acc_list,
                'gradient_norm': norm_list,
                'test loss': test_loss,
                'test_acc': test_acc
                }, os.path.join(save_path, 'r={}'.format(ratio)))


# ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# seed in [0, 1, 2, 3, 4]
seed = 0
ratio = 0.0
get_final_result(model, train_dataset, test_dataset, ratio, seed, save_path)
