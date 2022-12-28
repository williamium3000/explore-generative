import os

import torchvision
from torchvision.transforms import autoaugment, transforms

def build_trainset(args):
    if args.dataset == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif args.dataset == "mnist":
        return torchvision.datasets.MNIST(
            root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
            ]))
    elif args.dataset == "cifar100":
        return torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif args.dataset == "celeba":
        return torchvision.datasets.CelebA(
            root=args.data, split="train", download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    elif args.dataset == "imagenet":
        return torchvision.datasets.ImageFolder(
            root=os.path.join(args.data, "train"),
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(256),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))