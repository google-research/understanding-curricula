# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import shutil
import time
import warnings
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import get_dataset, get_model, get_optimizer, get_scheduler
from utils import LossTracker
import collections
import numpy as np
import subprocess
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--datadir', default='dataset',
                    help='path to dataset (default: dataset)')
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batchsize', default=256, type=int,
                    help='mini-batch size (default: 512), this is the total')
parser.add_argument('--optimizer', default="sgd", type=str,
                    help='optimizer')
parser.add_argument('--num_runs', default=10, type=int,
                    help='checkpoint model to resume')
parser.add_argument('--scheduler', default="cosine", type=str,
                    help='lr scheduler')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--printfreq', default=500, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--logdir', default='orders', type=str,
                    help='prefix to use when saving files')
parser.add_argument('--rand_fraction', default=0., type=float,
                    help='label curruption (default:0)')
args = parser.parse_args()

"""the code is based on  https://github.com/pluskid/structural-regularity"""
def main():
    tr_set = get_dataset(args.dataset, args.datadir, 'train',rand_fraction=args.rand_fraction)
    if args.dataset in ['cifar10', 'cifar100', 'cifar100N']:
        tr_set_clean = tr_set
    else:
        print ("ERROR: the dataset %s is not found "%(args.dataset))
    if 'cifar100N' == args.dataset:
        if args.rand_fraction == 0.2:
            name = 'cifar100N02'
        if args.rand_fraction == 0.4:
            name = 'cifar100N04'
        if args.rand_fraction == 0.6:
            name = 'cifar100N06'            
        if args.rand_fraction == 0.8:
            name = 'cifar100N08'
    else:
        name = args.dataset
        
    order = [i for i in range(len(tr_set))]
    ind_loss  = collections.defaultdict(list)
    for i_run in range(args.num_runs):
        random.shuffle(order)
        startIter = 0
        for i in range(4):
            if i==3:
                startIter_next = len(tr_set)
            else: 
                startIter_next = int(startIter+1/4*len(tr_set))
            print ('i_run %s and order  =============> from %s to %s'%(i_run, startIter,startIter_next))
            valsets = Subset(tr_set_clean, list(order[startIter:startIter_next]))
            trainsets = Subset(tr_set, list(order[0:startIter])+list(order[startIter_next:]))
            train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.workers, pin_memory=True) 
            val_loader = torch.utils.data.DataLoader(valsets, batch_size=args.batchsize,
                              shuffle=False, num_workers=args.workers, pin_memory=True) 
            
            ind_loss = subset_train(i_run,tr_set,train_loader,val_loader,list(order[startIter:startIter_next]),ind_loss,args)
            startIter += int(1/4*len(tr_set))
            
        stat = {k:[torch.mean(torch.tensor(v)),torch.std(torch.tensor(v))] for k, v in sorted(ind_loss.items(), key=lambda item:sum(item[1]))}
        if i_run == args.num_runs-1:
            torch.save(stat, os.path.join(args.logdir,name+'.order.pth'))
        else:
            torch.save(stat, os.path.join(args.logdir,name+'.order.'+str(i_run)+'.pth'))
        
    
def subset_train(seed,tr_set,train_loader,val_loader,val_order,ind_loss,args):
    set_seed(seed)
    model = get_model(args.arch, tr_set.nchannels, tr_set.imsize, len(tr_set.classes), False)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=args.epochs)
    start_epoch = 0
    
    for epoch in range(start_epoch, start_epoch+args.epochs):
        loss_acc = 0
        for i, (images, target) in enumerate(train_loader):
            images, target = cuda_transfer(images, target)
            output = model(images)
            loss = criterion(output, target)
            loss_acc += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
        print ('train at epoch %s with loss %f'%(epoch,loss_acc))
    return validate(val_loader, model, nn.CrossEntropyLoss(reduction="none").cuda(),val_order,ind_loss)

def validate(val_loader,model,criterion,val_order,ind_loss):
    # switch to evaluate mode
    model.eval()
    start = 0
    loss_acc = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images, target = cuda_transfer(images, target)
            output = model(images)
            indloss = criterion(output, target)
            list(map(lambda a, b : ind_loss[a].append(b), val_order[start:start+len(target)], indloss)) 
            start += len(target)
            loss_acc += torch.sum(indloss).item()
        print ('test with loss %f'%(loss_acc))
    return ind_loss

def set_seed(seed=None):
  if seed is not None:
      random.seed(seed)
      torch.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

def cuda_transfer(images, target):
  images = images.cuda(non_blocking=True)
  target = target.type(torch.LongTensor).cuda(non_blocking=True)
  return images, target


if __name__ == '__main__':
    main()

