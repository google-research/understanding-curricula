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
import wget
import time
import warnings
import json
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Subset

from utils import get_dataset, get_model, get_optimizer, get_scheduler
from utils import  LossTracker,run_cmd
from torch.utils.data import DataLoader
from utils import get_pacing_function,balance_order_val

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data-dir', default='dataset',
                    help='path to dataset')
parser.add_argument('--order-dir', default='cifar10-cscores-orig-order.npz',
                    help='path to train val idx')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture: (default: resnet18)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--printfreq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=128, type=int,
                    help='mini-batch size (default: 256), this is the total')
parser.add_argument('--optimizer', default="sgd", type=str,
                    help='optimizer')
parser.add_argument('--scheduler', default="cosine", type=str,
                    help='lr scheduler')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--half', default=False, action='store_true',
                    help='training with half precision')
# curriculum params
parser.add_argument("--pacing-f", default="linear", type=str, help="which pacing function to take")
parser.add_argument('--pacing-a', default=1., type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--pacing-b', default=1., type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument("--ordering", default="curr", type=str, help="which test case to use. supports: standard, curriculum, anti and random")
parser.add_argument('--rand-fraction', default=0., type=float,
                    help='label curruption (default:0)')
args = parser.parse_args()
def main():
    set_seed(args.seed) 
    # create training and validation datasets and intiate the dataloaders
    tr_set = get_dataset(args.dataset, args.data_dir, 'train',rand_fraction=args.rand_fraction)
    if args.dataset == "cifar100N":
        val_set = get_dataset("cifar100", args.data_dir, 'val')
        tr_set_clean = get_dataset("cifar100", args.data_dir, 'train')
    else:
        val_set = get_dataset(args.dataset, args.data_dir, 'val')        
    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,\
                              shuffle=True, num_workers=args.workers, pin_memory=True)  
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batchsize*2,
                      shuffle=False, num_workers=args.workers, pin_memory=True)

    criterion_ind = nn.CrossEntropyLoss(reduction="none").cuda()
    # initiate a recorder for saving and loading stats and checkpoints
    if  'cscores-orig-order.npz' in args.order_dir:
        temp_path = os.path.join("orders",args.dataset+'-cscores-orig-order.npz')
        if not os.path.isfile(temp_path):        
            print ('Downloading the data cifar10-cscores-orig-order.npz and cifar100-cscores-orig-order.npz to folder orders')
            if 'cifar100' == args.dataset:
                url = 'https://pluskid.github.io/structural-regularity/cscores/cifar100-cscores-orig-order.npz'
            if 'cifar10' == args.dataset:
                url = 'https://pluskid.github.io/structural-regularity/cscores/cifar10-cscores-orig-order.npz'
            wget.download(url, './orders')
        temp_x = np.load(temp_path)['scores']
        ordering = collections.defaultdict(list)
        list(map(lambda a, b: ordering[a].append(b), np.arange(len(temp_x)),temp_x))
        order = [k for k, v in sorted(ordering.items(), key=lambda item: -1*item[1][0])]
    else:
        print ('Please check if the files %s in your folder -- orders. See ./orders/README.md for instructions on how to create the folder' %(args.order_dir))
        order = [x for x in list(torch.load(os.path.join("orders",args.order_dir)).keys())]
        
    order,order_val = balance_order_val(order, tr_set, num_classes=len(tr_set.classes)) 
   
    #decide CL, Anti-CL, or random-CL
    if args.ordering == "random":
        np.random.shuffle(order)
    elif  args.ordering == "anti_curr":
        order = [x for x in reversed(order)]
      
    #check the statistics 
    bs = args.batchsize
    N = len(order)
    myiterations = (N//bs+1)*args.epochs
    
    #initial training
    model = get_model(args.arch, tr_set.nchannels, tr_set.imsize, len(tr_set.classes), args.half)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=myiterations)

    start_epoch = 0
    total_iter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],"test_loss": [], "test_acc": [], "iter": [0,] }
    start_time = time.time()
    
    if args.dataset == "cifar100N":
        val_set = Subset(tr_set_clean, order_val)
    else:
        val_set = Subset(tr_set, order_val)    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batchsize*2,
                              shuffle=False, num_workers=args.workers, pin_memory=True)                           
    trainsets = Subset(tr_set, order)
    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.workers, pin_memory=True) 
    criterion = nn.CrossEntropyLoss().cuda()

    if args.ordering == "standard":
        iterations = 0
        for epoch in range(args.epochs): 
            tr_loss, tr_acc1, iterations = train(train_loader, model, criterion, optimizer,scheduler, epoch,iterations)
            val_loss, val_acc1 = validate(val_loader, model, criterion)
            test_loss, test_acc1 = validate(test_loader, model, criterion)                 
            print ("%s epoch %s iterations w/ LEARNING RATE %s"%(epoch, iterations,optimizer.param_groups[0]["lr"])) 
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)                           
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc1)  
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc1)
            history["iter"].append(iterations)
            torch.save(history,"stat.pt")  
    else:
        all_sum = N/(myiterations*(myiterations+1)/2)
        iter_per_epoch = N//bs         
        pre_iterations = 0
        startIter = 0
        pacing_function = get_pacing_function(myiterations, N, args)
            
        startIter_next = pacing_function(0) # <=======================================
        print ('0 iter data between %s and %s w/ Pacing %s'%(startIter,startIter_next,args.pacing_f,))
        trainsets = Subset(tr_set, list(order[startIter:max(startIter_next,256)]))
        train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.workers, pin_memory=True) 
        dataiter = iter(train_loader)
        step = 0
        
        while step < myiterations:   
            tracker = LossTracker(len(train_loader), f'iteration : [{step}]', args.printfreq)
            for images, target in train_loader:
                step += 1
                images, target = cuda_transfer(images, target)
                output = model(images)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                tracker.update(loss, output, target)
                tracker.display(step-pre_iterations)

            #If we hit the end of the dynamic epoch build a new data loader
            pre_iterations = step          
            if startIter_next <= N:            
                startIter_next = pacing_function(step)# <=======================================
                print ("%s iter data between %s and %s w/ Pacing %s and LEARNING RATE %s "%(step,startIter,startIter_next,args.pacing_f, optimizer.param_groups[0]["lr"]))
                train_loader = torch.utils.data.DataLoader(Subset(tr_set, list(order[startIter:max(startIter_next,256)])),\
                                                           batch_size=args.batchsize,\
                                                           shuffle=True, num_workers=args.workers, pin_memory=True)
            # start your record
            if step > 50: 
                tr_loss, tr_acc1 = tracker.losses.avg, tracker.top1.avg 
                val_loss, val_acc1 = validate(val_loader, model, criterion) 
                test_loss, test_acc1 = validate(test_loader, model, criterion)                       
                # record
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc1)                                       
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc1)                 
                history["train_loss"].append(tr_loss)
                history["train_acc"].append(tr_acc1)  
                history['iter'].append(step) 
                torch.save(history,"stat.pt")  
                # reinitialization<=================
                model.train()
                
        
def train(train_loader, model, criterion, optimizer,scheduler, epoch, iterations):
  # switch to train mode
  model.train()
  tracker = LossTracker(len(train_loader), f'Epoch: [{epoch}]', args.printfreq)
  for i, (images, target) in enumerate(train_loader):
    iterations += 1
    images, target = cuda_transfer(images, target)
    output = model(images)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    tracker.update(loss, output, target)
    tracker.display(i)
    scheduler.step()
  return tracker.losses.avg, tracker.top1.avg,  iterations

def validate(val_loader, model, criterion):
  # switch to evaluate mode
  model.eval()
  with torch.no_grad():
    tracker = LossTracker(len(val_loader), f'val', args.printfreq)
    for i, (images, target) in enumerate(val_loader):
      images, target = cuda_transfer(images, target)
      output = model(images)
      loss = criterion(output, target)
      tracker.update(loss, output, target)
      tracker.display(i)
  return tracker.losses.avg, tracker.top1.avg

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

