import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.nn as nn
from train_test import train, test
from image_getdata import getdata
from networks import ResNet18, mlp, Wide_ResNet


train_loader, test_loader = getdata('CIFAR10', train_bs=64, test_bs=100)
model = ResNet18(num_classes=10).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
save_path = './result/ResNet18/SGD'
id = str(np.random.uniform())

# train_loader, test_loader = getdata('CIFAR10', train_bs=64, test_bs=100)
# model = Wide_ResNet(depth=16, widen_factor=8, dropout_rate=0.0, num_classes=10).cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# save_path = './result/Wide_ResNet/SGD'
# id = str(np.random.uniform())

# train_loader, test_loader = getdata('MNIST', train_bs=64, test_bs=100)
# model = mlp(layer=2, use_bn=False, bias=False, act='ReLU', dims=[28*28, 100, 10]).cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# save_path = './result/2_MLP/SGD'
# id = str(np.random.uniform())


criterion = nn.CrossEntropyLoss()
error = 1
epoch = 0
train_loss_list, train_acc_list, norm_list = [], [], []
while error > 1e-3 and epoch < 1000:
    train_loss, train_acc, norm = train(model, train_loader, criterion, optimizer)
    error = train_loss
    epoch += 1
    print('epoch: {}, training loss: {}, acc: {}, gradient norm: {}'.format(epoch, train_loss, train_acc, norm))
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    norm_list.append(norm)


test_loss, test_acc = test(model, test_loader, criterion)
torch.save({
            'train_loss': train_loss_list,
            'train_acc': train_acc_list,
            'gradient_norm': norm_list,
            'test loss': test_loss,
            'test_acc': test_acc
            }, os.path.join(save_path, id))
