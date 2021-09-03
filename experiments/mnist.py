#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:34:06 2021

@author: angusphillips
"""

# =============================================================================
# Import Libraries//General preparation
# =============================================================================

import numpy as np
import pandas as pd
import random
import torch
import torch.optim as optim
from pbms.data import DATA
from pbms.models import FCN, ProbFCN
from pbms.methods import trainNN, testNN, trainProbNN, testProbNN, compute_bounds_and_metrics, count_parameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Reproducibility
torch.manual_seed(2)
np.random.seed(2)
random.seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# Experimental set-up
# =============================================================================

not_my_variables = set(dir())

# Fixed parameters
dataset = 'mnist'
prior_percentage = 0.7
prior_epochs = 100
train_epochs = 50
batch_size = 256
optimizer = 'Adam'
pmin = 1e-4
delta = 0.025
delta_p = 0.01
dropout = 0.2
sigma_prior = 0.01
mc_samples = 20000
ensemble_samples = 100

# Gridsearch parameters
depth = [1, 2, 3, 4]
width = [20, 50, 100, 200, 500]

n_models = len(depth) * len(width)
prior_weight = 1/n_models

# Set up for saving results
path = 'RESULTS/MNIST/'
save = True

# Save an experiment description
all_variables = set(dir())
my_variables = all_variables - not_my_variables
if save:
    # Store experiment description
    file = open(path + 'description.txt', 'w')
    for var in list(my_variables):
        if var != 'not_my_variables' and not(var.startswith('_')):
            line = var + " = " + str(eval(var))
            file.write(line + "\n")
    file.close()
    
# Dataframes for storing results and training metrics
metric_names = ['architecture', 'depth', 'width', 'n_parameters', 'loss_bound', 'error_bound', 'stch_loss', 'stch_err', 'post_loss', 'post_err', 'ens_loss', 'ens_err', 'train_ens_err', 'prior_err', 'prior_train_err']
results = pd.DataFrame(np.zeros((n_models, len(metric_names))), columns = metric_names)
prior_training_metrics = pd.DataFrame(np.zeros((n_models, prior_epochs)))
prior2_training_metrics = pd.DataFrame(np.zeros((n_models*2, prior_epochs)))
prob_training_metrics = pd.DataFrame(np.zeros((n_models*2, train_epochs)))

counter = 0

# =============================================================================
# Data
# =============================================================================

data = DATA(dataset, prior_percentage, batch_size)
image_size = data.image_size
n_classes = data.n_classes

# =============================================================================
# Experiment
# =============================================================================

for d in depth:
    for w in width:
            print('\nWorking on model {} of {}'.format(counter+1, n_models))
            
            ######## Prior means network ########
            # Train model
            prior = FCN(image_size, n_classes, w, d, dropout).to(device)
            prior_optimizer = optim.Adam(prior.parameters())
            loss_progress = trainNN(prior, prior_optimizer, prior_epochs, data.prior_loader, pmin, device, verbose=False)
            
            # Store training metrics
            prior_training_metrics.iloc[counter, :] = loss_progress
            
            if save: # Save model
                model_name = f"MODELS/prior_{counter:d}"
                torch.save(prior.state_dict(), path + model_name + '.pt')
                
            ######## Prior variance network ########
            # Train model
            prior2 = ProbFCN(image_size, n_classes, w, d, sigma_prior, True, prior, device).to(device)
            prior_optimizer2 = optim.Adam(prior2.parameters())
            obj_progress, loss_progress, prior_var_progress = trainProbNN(prior2, prior_optimizer2, 'classic', True, prior_epochs, data.prior_loader, delta, pmin, prior_weight, device, verbose=False)
            
            # Store training metrics
            prior2_training_metrics.iloc[2*counter, :] = loss_progress
            prior2_training_metrics.iloc[2*counter + 1, :] = obj_progress
            
            if save: # Save model
                model_name = f"MODELS/prior2_{counter:d}"
                torch.save(prior2.state_dict(), path + model_name + '.pt')
                pd.DataFrame(prior_var_progress).to_csv(path + f'prior_var_{counter:d}.csv')
                        
            
            ######## Probabilistic network ########
            prob = ProbFCN(image_size, n_classes, w, d, sigma_prior, False, prior2, device).to(device)
            prob_optimizer = optim.Adam(prob.parameters())
            loss_progress, obj_progress = trainProbNN(prob, prob_optimizer, 'classic', False, train_epochs, data.train_loader, delta, pmin, prior_weight, device, verbose=False)
    
            prob_training_metrics.iloc[2*counter, :] = loss_progress
            prob_training_metrics.iloc[2*counter + 1, :] = obj_progress
            
            if save:
                model_name = f'MODELS/prob_{counter:d}'
                torch.save(prob.state_dict(), path + model_name + '.pt')
    
            ######## Evaluation ########
            _, error_prior = testNN(prior, data.test_loader, pmin, device)
            _, train_err_prior = testNN(prior, data.train_loader_whole, pmin, device)
            stch_loss, stch_err = testProbNN(prob, 'stochastic', data.test_loader, ensemble_samples, pmin, device)
            post_loss, post_err = testProbNN(prob, 'posterior-mean', data.test_loader, ensemble_samples, pmin, device)
            ens_loss, ens_err = testProbNN(prob, 'ensemble', data.test_loader, ensemble_samples, pmin, device)
            _, train_ens_err = testProbNN(prob, 'ensemble', data.train_loader_whole, ensemble_samples, pmin, device)
            ce_bound, error_bound, _, _, _ = compute_bounds_and_metrics(prob, data.bound_loader_whole, 'classic', delta, delta_p, mc_samples, pmin, prior_weight, device)
            n_parameters = count_parameters(prob)
            
            results.iloc[counter, :] = ['FCN', d, w, n_parameters, float(ce_bound), float(error_bound), stch_loss, stch_err, post_loss, post_err, ens_loss, ens_err, train_ens_err, error_prior, train_err_prior]
            
            counter += 1
            
            if save:
                results.to_csv(path + 'results_mnist.csv')
                prior_training_metrics.to_csv(path + 'prior_training_metrics.csv')
                prior2_training_metrics.to_csv(path + 'prior2_training_metrics.csv')
                prob_training_metrics.to_csv(path + 'prob_training_metrics.csv')
    
    
    
    