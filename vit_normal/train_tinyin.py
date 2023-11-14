# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

from datasets import load_dataset, Image

import os
import argparse
import pandas as pd
import csv
import pickle
import time
import cv2 as cv
from PIL import Image

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT

# parsers
parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--dataset', default='cifar10', type=str)

args = parser.parse_args()

# take in args

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes=200
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
print('==> Preparing data..')
if args.net == "vit_timm":
    size = 384
else:
    size = imsize



tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
dataset_val = load_dataset('Maysee/tiny-imagenet', split='validation')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def transformsTrain(examples):

    transformed_examples = {}

    transformed_examples["pixel_values"] = [transform_train(image) for image in examples["image"]]
    transformed_examples["labels"] = examples["label"]  # Assuming the key for labels is "label," please adjust accordingly

    return transformed_examples




dataset_train = tiny_imagenet_train.with_transform(transformsTrain)


def collate_fn(examples):

    images = []

    labels = []

    for example in examples:

        images.append((example["pixel_values"]))

        labels.append(example["labels"])

        

    pixel_values = torch.stack(images)

    labels = torch.tensor(labels)

    return {"pixel_values": pixel_values, "labels": labels}

dataloaderTrain = DataLoader(dataset_train, collate_fn=collate_fn, batch_size=4)
dataloaderVal = DataLoader(dataset_val, collate_fn=collate_fn, batch_size=4)


'''
# Prepare dataset
tiny_imagenet_path_train = '../../tinyimagenet-ViT/TinyImageNet-Transformers/train'
tiny_imagenet_path_val = '../../tinyimagenet-ViT/TinyImageNet-Transformers/val'
tiny_imagenet_path_test = '../../tinyimagenet-ViT/TinyImageNet-Transformers/test'

tiny_imagenet_train_dataset = TinyImageNetDataset(root_dir=tiny_imagenet_path_train, transform=transform_train)
tiny_imagenet_val_dataset = TinyImageNetDataset(root_dir=tiny_imagenet_path_val, transform=transform_test)
tiny_imagenet_test_dataset = TinyImageNetDataset(root_dir=tiny_imagenet_path_test, transform=transform_test)

print(len(tiny_imagenet_train_dataset))

# Split trainset into train and validation
tiny_imagenet_trainset, tiny_imagenet_valset = torch.utils.data.random_split(tiny_imagenet_train_dataset, [80000, 20000])

# Dataloader
trainloader = torch.utils.data.DataLoader(tiny_imagenet_trainset, batch_size=bs, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(tiny_imagenet_valset, batch_size=bs, shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(tiny_imagenet_test_dataset, batch_size=bs, shuffle=False, num_workers=8)
'''
classes = [str(i) for i in range(200)]


# Model factory..
print('==> Building model..')

# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
# elif args.net=='vgg':
#     net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="vit_s":
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = 384,
    depth = 12,
    heads = 6,
    mlp_dim = 1536,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_ti":
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 12,
    heads = 3,
    mlp_dim = 768,
    dropout = 0.1,
    emb_dropout = 0.1
)
# elif args.net=="vit_small":
#     from models.vit_small import ViT
#     net = ViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = num_classes,
#     dim = 384,
#     depth = 12,
#     heads = 6,
#     mlp_dim = 1536,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# elif args.net=="simplevit":
#     from models.simplevit import SimpleViT
#     net = SimpleViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512
# )
# elif args.net=="vit":
#     # ViT for cifar10
#     net = ViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# elif args.net=="vit_timm":
#     import timm
#     net = timm.create_model("vit_base_patch16_384", pretrained=True)
#     net.head = nn.Linear(net.head.in_features, 10)
# elif args.net=="cait":
#     from models.cait import CaiT
#     net = CaiT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 6,   # depth of transformer for patch to patch attention only
#     cls_depth=2, # depth of cross attention of CLS tokens to patch
#     heads = 8,
#     mlp_dim = 512,
#     dropout = 0.1,
#     emb_dropout = 0.1,
#     layer_dropout = 0.05
# )

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloaderTrain):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloaderTrain), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
    return train_loss/(batch_idx+1), acc

##### Validation
def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloaderVal):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloaderVal), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {val_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return val_loss/(batch_idx+1), acc

'''
def test(epoch):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return test_loss/(batch_idx+1), acc
'''

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = val(epoch)
    #test_loss, test_acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    print('epoch:', epoch )
    print('train_loss: ', train_loss)
    print("train_acc", train_acc )
    print('val_loss', val_loss )
    print('val_acc', val_acc )

    print('epoch:', epoch )
    print( "lr:", optimizer.param_groups[0]["lr"])



    # Log training..


    # Write out csv..

# writeout wandb

    
