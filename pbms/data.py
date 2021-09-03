#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:49:53 2021

@author: angusphillips
"""

import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

class DATA():
    """
    Handles the data. Puts n data points into main training set,
    n_0 data points for learning the prior and n-n_0 data points for evaluating
    the bound. Also creates the usual test set. The prior and training loaders 
    use the given batch size whereas the bound loaders and test loaders are in
    one batch. (If this doesn't fit in memory, some adjustment will be required
    which can be performed for each different dataset.)
    
    Parameters
    ----------
    dataset : string
        Name of dataset to import. Options: 'mnist', 'fmnist', 'cifar10', 'cifar100'
    perc_prior : float or None
        Percentage of training data to learn prior (expressed as float in (0, 1))
    batch_size : int
        Batch size
    """
    
    def __init__(self, dataset, perc_prior, batch_size):
        
        self.batch_size = batch_size
        
        if dataset == 'mnist':
            # Transformer for pre-processing MNIST data
            transformer = T.Compose(
                    [T.ToTensor(),
                     T.Normalize((0.1307,), (0.3081,))])
            self.train = datasets.MNIST('data/', train = True, download = True,
                                           transform = transformer)
            self.test = datasets.MNIST('data/', train = False, download = True,
                                       transform = transformer)
            
        elif dataset == 'fmnist':
            transformer = T.Compose(
                [T.ToTensor(),
                 T.Normalize((0.2860,), (0.3530,))])
            self.train = datasets.FashionMNIST('data/', train = True, download = True,
                                               transform = transformer)
            self.test = datasets.FashionMNIST('data/', train = False, download = True,
                                              transform = transformer)
            
        elif dataset == 'cifar10':
            transformer = T.Compose(
                [T.ToTensor(),
                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            self.train = datasets.CIFAR10('data/', train = True, download = True,
                                        transform = transformer)
            self.test = datasets.CIFAR10('data/', train = False, download = True,
                                       transform = transformer)        
            
        elif dataset == 'cifar100':
            transformer = T.Compose(
                [T.ToTensor(),
                 T.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023))])
            self.train = datasets.CIFAR100('data/', train = True, download = True,
                                           transform = transformer)
            self.test = datasets.CIFAR100('data/', train = True, download = True,
                                          transform = transformer)
        
        else:
            print('Invalid dataset {}'.format(dataset))
            
        # Data Attributes
        self.image_size = self.train[0][0].size()
        self.n_classes = len(self.train.classes)
        self.n_train = len(self.train)
        self.n_test = len(self.test)
            
        if perc_prior:
            # Splitting the training data into prior and bound data
            self.n_prior = int(np.round(self.n_train * perc_prior))
            self.n_bound =self.n_train - self.n_prior
            self.prior, self.bound = random_split(self.train, (self.n_prior, self.n_bound))
            
            # Form dataloaders
            self.prior_loader = DataLoader(dataset = self.prior, shuffle = True, batch_size = batch_size)
            self.train_loader = DataLoader(dataset = self.train, shuffle = True, batch_size = batch_size)
            self.bound_loader = DataLoader(dataset = self.bound, shuffle = True, batch_size = batch_size)
            if dataset == 'fmnist' or dataset == 'cifar10' or dataset == 'cifar100':
                self.bound_loader_whole = DataLoader(dataset = self.bound, shuffle = True, batch_size = self.n_bound//2)
                self.train_loader_whole = DataLoader(dataset = self.train, shuffle = True, batch_size = self.n_train//8)
            else:
                self.bound_loader_whole = DataLoader(dataset = self.bound, shuffle = True, batch_size = self.n_bound)
                self.train_loader_whole = DataLoader(dataset = self.train, shuffle = True, batch_size = self.n_train)
            self.test_loader = DataLoader(dataset = self.test, shuffle = True, batch_size = self.n_test)
            
            
        else:
            self.train_loader = DataLoader(dataset = self.train, shuffle = True, batch_size = batch_size)
            self.test_loader = DataLoader(dataset = self.test, shuffle = True, batch_size = batch_size)