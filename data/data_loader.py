from torchvision.datasets import SVHN, CIFAR10, MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_loader(opt):
    print('=========================data_loader.py-->get_data_loader(opt)')
    if opt.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if opt.is_train:
            train_set = CIFAR10(root='./data', train=True, transform=transform, download=True)
            return DataLoader(
                train_set,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_preprocess_workers
            )
        else:
            test_set = CIFAR10(root='./data', train=False, transform=transform, download=True)
            return DataLoader(
                test_set,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_preprocess_workers
            )
    if opt.dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = SVHN(root='./data', transform=transform, download=True)
        if opt.is_train:
            return DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_preprocess_workers
            )
        else:
            testset = SVHN(root='./data', split='test', transform=transform, download=True)
            print(testset)
            return DataLoader(
                testset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_preprocess_workers
            )
    if opt.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if opt.is_train:
            dataset = MNIST(root='./data', train=True, transform=transform, download=True)
            return DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_preprocess_workers
            )
        else:
            testset = MNIST(root='./data', train=False, transform=transform, download=True)
            print(testset)
            return DataLoader(
                testset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_preprocess_workers
            )
