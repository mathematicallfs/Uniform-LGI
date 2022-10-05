import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.nn as nn
from train_test import train
from networks import mlp



# create custom dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class CustomDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label
    def __len__(self): return len(self.label)
    def __getitem__(self, idx):
        x = self.input[idx,:]
        y = self.label[idx,:]
        return x, y


n = 100
d = 200
beta = torch.normal(0, 1, size=(d, 1))
beta = beta / torch.linalg.norm(beta)
train_input = torch.normal(0, 1, size=(n, d))
train_input = train_input / torch.linalg.norm(train_input, dim=1, keepdim=True)
train_label = train_input@beta
train_data = CustomDataset(train_input, train_label)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
save_path = './result/Linear_regression'



def get_result(model, train_loader, criterion):
    epoch = 0
    train_loss_list, norm_list = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    while epoch < 100000:
        train_loss, _, norm = train(model, train_loader, criterion, optimizer)
        epoch += 1
        print('epoch: {}, training loss: {}, gradient norm: {}'.format(epoch, train_loss, norm))
        train_loss_list.append(train_loss)
        norm_list.append(norm)      
    id = str(np.random.uniform())
    torch.save({
                'train_loss': train_loss_list,
                'gradient_norm': norm_list,
                }, os.path.join(save_path, id))



for i in range(10):
    model = mlp(layer=1, use_bn=False, bias=False, act='Linear', dims=[d, 1]).cuda()
    criterion = nn.MSELoss()
    get_result(model, train_loader, criterion)
