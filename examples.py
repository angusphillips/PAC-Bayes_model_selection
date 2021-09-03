#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:31:07 2021

@author: angusphillips
"""

import numpy as np
import random
import torch
import torch.optim as optim

from pbms.data import DATA
from pbms.models import FCN, ProbFCN, CNN, ProbCNN, ResNet, ProbResNet
from pbms.methods import trainNN, trainProbNN, count_parameters, compute_bounds_and_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Reproducibility
torch.manual_seed(2)
np.random.seed(2)
random.seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

not_my_variables = set(dir())

# Experiment Parameters
prior_epochs = 50
train_epochs = 20
prior_percentage = 0.7
optimizer = 'Adam'
batch_size = 256
sigma_prior = 0.1
delta = 0.025
delta_p = 0.01
pmin = 1e-4
mc_samples = 100000
ensemble_samples = 100

# =============================================================================
# FCN
# =============================================================================

width = 0
depth = 20
dropout = 0.2

# Import data
data = DATA('cifar10', prior_percentage, batch_size)
image_size = data.image_size
n_classes = data.n_classes

# Learn prior means by ERM
priorFCN = FCN(image_size, n_classes, width, depth, dropout).to(device)
prior_optimizer = optim.Adam(priorFCN.parameters())
loss_progress = trainNN(priorFCN, prior_optimizer, prior_epochs, data.prior_loader, pmin, device, verbose=False)

# Learn prior variance
priorFCN2 = ProbFCN(image_size, n_classes, width, depth, sigma_prior, True, priorFCN, device).to(device)
prior_optimizer2 = optim.Adam(priorFCN2.parameters())
obj_progress, loss_progress, prior_var_progress = trainProbNN(priorFCN2, prior_optimizer2, 'classic', True, prior_epochs, data.prior_loader, delta, pmin, 1, device, verbose=False)

# Initialise probabilistic network and train
probFCN = ProbFCN(image_size, n_classes, width, depth, sigma_prior, False, priorFCN2, device).to(device)
optimizer = optim.Adam(probFCN.parameters())
obj_progress, loss_progress = trainProbNN(probFCN, optimizer, 'classic', False, train_epochs, data.train_loader, delta, pmin, 1, device, verbose=False)


# =============================================================================
# CNN
# =============================================================================

depth = 3
width = 32

# Import data
data = DATA('fmnist', prior_percentage, batch_size)
image_size = data.image_size
n_classes = data.n_classes

# Learn prior means by ERM
priorCNN = CNN(image_size, n_classes, width, depth, 3, 1, device).to(device)
prior_optimizer = optim.Adam(priorCNN.parameters())
loss_progress = trainNN(priorCNN, prior_optimizer, prior_epochs, data.prior_loader, pmin, device, verbose=False)

# Learn prior variance
priorCNN2 = ProbCNN(image_size, n_classes, width, depth, 3, 1, sigma_prior, True, priorCNN, device).to(device)
prior_optimizer2 = optim.Adam(priorCNN2.parameters())
obj_progress, loss_progress, prior_var_progress = trainProbNN(priorCNN2, prior_optimizer2, 'classic', True, prior_epochs, data.prior_loader, delta, pmin, 1, device, verbose=False)

# Initialise probabilistic network and train
probCNN = ProbCNN(image_size, n_classes, width, depth, 3, 1, sigma_prior, False, priorCNN2, device).to(device)
optimizer = optim.Adam(probCNN.parameters())
obj_progress, loss_progress = trainProbNN(probCNN, optimizer, 'classic', False, train_epochs, data.train_loader, delta, pmin, 1, device, verbose=False)

ce_bound, error_bound, _, _, _ = compute_bounds_and_metrics(probCNN, data.bound_loader_whole, 'classic', delta, delta_p, mc_samples, pmin, 1, device)


# =============================================================================
# ResNet
# =============================================================================

width = 8
blocks = [8, 8, 8, 8]

# Learn prior means by ERM
priorResNet = ResNet(image_size, n_classes, width, blocks).to(device)
prior_optimizer = optim.SGD(priorResNet.parameters(), lr = 0.01, momentum = 0.9)
# TODO: Make ResNet models take pmin in the forward routine
loss_progress = trainNN(priorResNet, prior_optimizer, prior_epochs, data.prior_loader, device, verbose=False)

# Learn prior variance
priorResNet2 = ProbResNet(image_size, n_classes, width, blocks, sigma_prior, True, priorResNet, device).to(device)
optimizer = optim.SGD(priorResNet2.parameters(), lr = 0.01, momentum = 0.9)
obj_progress, loss_progress, prior_var_progress = trainProbNN(priorResNet2, optimizer, 'classic', True, prior_epochs, data.prior_loader, delta, pmin, 1, device, verbose=False)

# Initialise probabilistic network and train
probResNet = ProbResNet(image_size, n_classes, width, blocks, sigma_prior, False, priorResNet2, device).to(device)
optimizer = optim.SGD(probResNet.parameters(), lr = 0.01, momentum = 0.9)
obj_progress, loss_progress = trainProbNN(probResNet, optimizer, 'classic', False, train_epochs, data.train_loader, delta, pmin, 1, device, verbose=False)










