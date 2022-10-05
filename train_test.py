import torch
import math

# training with specific epochs

def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for _, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        if type(model).__name__ == 'mlp':
            input = input.reshape((input.shape[0], -1))
        
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        train_loss += loss.item()

    # get the gradient norm, excluding bn
    blacklist = {'bn'}
    total_norm = 0
    for name, p in model.named_parameters():
        if all(x not in name for x in blacklist):
            total_norm = total_norm + (torch.norm(p.grad.data).item())**2
    total_norm = math.sqrt(total_norm)
    
    train_loss = train_loss / len(train_loader)
    acc = correct / total
    
    return train_loss, acc, total_norm




def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.cuda(), target.cuda()

            # only for MLP
            input = input.reshape((input.shape[0], -1))
            
            output = model(input)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss = test_loss / len(test_loader)
    acc = correct / total
    
    return test_loss, acc
