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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import get_dataset, get_model, get_optimizer, get_scheduler
from utils import LossTracker
from third_party.data import DataLoader
import collections
import numpy as np
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--data-dir', default='dataset',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total')
parser.add_argument('--optimizer', default="sgd", type=str,
                    help='optimizer')
parser.add_argument('--scheduler', default="cosine", type=str,
                    help='lr scheduler')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--printfreq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save-fre', default=14, type=int,
                    help='save the frequency. ')
parser.add_argument('--half', default=False, action='store_true',
                    help='training with half precision')
parser.add_argument('--logdir', default='', type=str,
                    help='prefix to use when saving files')
parser.add_argument('--saveparam', default='optimizer', type=str,
                    help='param names and values to use when saving files')
args = parser.parse_args()

def main():

    set_seed(args.seed)
    # create training and validation datasets and intiate the dataloaders
    tr_set = get_dataset(args.dataset, args.data_dir, 'train')
    val_set = get_dataset(args.dataset, args.data_dir, 'val')

    train_loader = DataLoader(tr_set, batch_size=args.batchsize,
                      shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batchsize,
                      shuffle=False, num_workers=args.workers, pin_memory=True)
    
    model = get_model(args.arch, tr_set.nchannels, tr_set.imsize, len(tr_set.classes), args.half)
    # define loss function (criterion), optimizer and scheduler
    criterion = nn.CrossEntropyLoss(reduction="none").cuda()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=args.epochs)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "iter": [0,] }
    #start_epoch = rc.resume_full_checkpoint(args.resume, model, optimizer, scheduler)
    start_epoch=0
    total_iter = 0
    if args.evaluate:
        validate(val_loader, model, criterion,total_iter)
    else:
        for epoch in range(start_epoch, args.epochs):
            tr_loss, tr_acc1, tr_acc5, val_loss, val_acc1, val_acc5, total_iter = train_eval(train_loader,val_loader, model, criterion, optimizer, epoch,total_iter)
            scheduler.step()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc1)  
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc1)
            history["iter"].append(total_iter)
            torch.save(history,"stat.pt")  

def train_eval(train_loader,val_loader, model,criterion, optimizer, epoch, total_iter):
    ind_train  = collections.defaultdict(list)
    ind_test  = collections.defaultdict(list)
    # switch to train mode
    model.train()
    tracker = LossTracker(len(train_loader), f'Epoch: [{epoch}]', args.printfreq)
    for i, ((images, target),index) in enumerate(train_loader):

        images, target = cuda_transfer(images, target)
        output = model(images)

        # RECORD
        confidence = F.softmax(output)
        pred = torch.max(confidence, 1)[-1]
        indloss = criterion(output, target)
        list(map(lambda a, b, c: ind_train[a].append([b.item(),c]), index, indloss,pred.view(-1).tolist()))
        # END RECORD
        
        loss = sum(indloss)/len(target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #tracking acc and loss
        tracker.update(loss, output, target)
        tracker.display(i)
        
        if i%args.save_fre == (args.save_fre-1) or (i+1)*args.batchsize>=50000:
            val_loss, val_acc1, val_acc5, ind_test = validate(val_loader, model, criterion, ind_test)
            model.train()
    torch.save(ind_test,  os.path.join(args.logdir, "test-"+str(epoch)+'.pt'))
    torch.save(ind_train, os.path.join(args.logdir, "train-"+str(epoch)+'.pt'))   
    return tracker.losses.avg, tracker.top1.avg, tracker.top5.avg, val_loss, val_acc1, val_acc5,(epoch+1)*(i+1)

def validate(val_loader, model, criterion,ind_test):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        tracker = LossTracker(len(val_loader), f'Test', args.printfreq)
        for i, ((images, target),index) in enumerate(val_loader):

            images, target = cuda_transfer(images, target)
            output = model(images)
            # RECORD
            confidence = F.softmax(output)
            pred = torch.max(confidence, 1)[-1]
            indloss = criterion(output, target) 
            list(map(lambda a, b,c: ind_test[a].append([b.item(),c]), index, indloss,pred.view(-1).tolist()))
            # END
            loss = sum(indloss)/len(target)
            tracker.update(loss, output, target)
            tracker.display(i)

    return tracker.losses.avg, tracker.top1.avg, tracker.top5.avg, ind_test

def set_seed(seed=None):
    if seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')
def cuda_transfer(images, target):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    if args.half: images = images.half()
    return images, target


if __name__ == '__main__':
    main()

