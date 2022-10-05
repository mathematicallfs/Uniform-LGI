from torchvision import transforms, datasets
import torch.utils.data


def getdata(name, train_bs, test_bs):
    
    path = './data'
    
    if name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_test = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
        test_dataset = datasets.CIFAR10(path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)
        
        
    if name == 'MNIST':
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
        test_dataset= datasets.MNIST(path, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)


    return train_loader, test_loader
