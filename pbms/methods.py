#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:05:50 2021

@author: angusphillips
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pbms.bounds import compute_bound, inv_kl

# =============================================================================
# Non-Probabilistic Methods
# =============================================================================

def trainNN(net, optimizer, epochs, train_loader, pmin, device, verbose=False):
    """
    Train function for a standard NN. Objective is the bounded cross entropy loss.

    Parameters
    ----------
    net : nn object
        Network object to train
    optimizer : torch.optim object
        Optimizer to use in training
    epochs : int
        Number of epochs to train for
    train_loader: DataLoader object
        Train loader to use for training
    pmin : float
        Minimum probability used to upper bound CE loss
    device : string
        Device the code will run in (e.g. 'cuda')
    verbose : bool
        Whether to print training metrics
    """
    net.train()
    n_batches = len(train_loader)
    loss_progress = np.zeros(epochs)
    for epoch in tqdm(range(epochs), position=0, leave=True):
        # CUMULATIVE training metrics
        c_total, c_correct, c_loss = 0.0, 0.0, 0.0
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            if hasattr(net, 'RN'):
                output = net(data)
            else:
                output = net(data, pmin)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            c_correct += pred.eq(target.view_as(pred)).sum().item()
            c_total += target.size(0)
            c_loss += loss.detach()
        loss_progress[epoch] = c_loss/n_batches
        if verbose:
            print(f"-Epoch {epoch + 1 :.5f}, Train loss (per batch): {c_loss/n_batches :.5f}, Train error:  {1-(c_correct/c_total):.5f}")
    
    return loss_progress
   
        
def testNN(net, test_loader, pmin, device):
    """
    Test function for a standard NN. Reports the bounded cross-entropy loss
    and the 0-1 error rate.

    Parameters
    ----------
    net : nn object
        Network object to test
    test_loader : DataLoader object
        Test data loader
    pmin : float
        Minimum probability used to upper bound CE loss
    device : string
        Device the code will run in (e.g. 'cuda')
    """
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            if hasattr(net, 'RN'):
                outputs = net(data)
            else:
                outputs = net(data, pmin)
            loss += F.nll_loss(outputs, target)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    n_batches = len(test_loader)
    ce_loss = loss/n_batches
    error = 1-(correct/total)
    
    # Note the loss returned here is the unbounded version
    return ce_loss, error


# =============================================================================
# Probabilistic Methods
# =============================================================================

def trainProbNN(net, optimizer, objective, learn_prior, epochs, train_loader, delta, pmin, prior_weight, device, verbose=False):
    """
    Train an instance of a probabilistic NN. Used for both stage 2 and 
    stage 3 training depending on the value of learn_prior. 
    
    Parameters
    ----------
    net : nn.Module
        Neural network object to train
    optimizer : torch.optim
        Optimizer to use for training eg SGD/ADAM
    objective : string
        The type of training objective: currently 'classic'
    learn_prior : bool
        Indicates whether the prior variance should be learnt or not
    epochs : int
        Number of epochs to train for
    train_loader : DataLoader
        Training data  
    delta : float, optional
        Tolerance in the PB bound
    pmin : float, optional
        Minimum probability used to upper bound CE loss
    prior_weight : float
        prior weight which appears in the modified PB bound
    device : string
        Where to run code.
    verbose : bool
        Whether to print training metrics
    """
    net.train()
    
    # Lists for tracking convergence
    n_batches = len(train_loader)
    obj_progress = np.zeros(epochs)
    loss_progress = np.zeros(epochs)
    if learn_prior:
        if hasattr(net, 'blocks'):
            n_layers = net.depth
        else:
            n_layers = len(net.layers)
        prior_sigma_progress = np.zeros((n_layers, epochs))
    
    for epoch in tqdm(range(epochs), position=0, leave=True):
        # variables to keep CUMULATIVE training metrics
        c_err, c_bound, c_kl, c_loss = 0.0, 0.0, 0.0, 0.0
        if learn_prior:
            sigma = np.zeros(n_layers)
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            ce_loss, err, kl = compute_losses(net, pmin, data = data, target = target)
            bound = compute_bound(ce_loss, kl, len(train_loader.dataset), delta, objective, prior_weight)
            bound.backward()
            optimizer.step()
            c_bound += bound.item()
            c_kl += kl.item()
            c_loss += ce_loss.item()
            c_err += err
            if learn_prior:
                if hasattr(net, 'blocks'):
                    sigma += ResNet_extractSigma(net)
                else:
                    sigma += [float(F.softplus(net.layers[l].distributions.rho.detach())) for l in range(n_layers)]
        obj_progress[epoch] = c_bound/n_batches
        loss_progress[epoch] = c_loss/n_batches
        if learn_prior:
            prior_sigma_progress[:, epoch] = sigma/n_batches
        if verbose:
            # show the batch average metrics for given epoch
            print(f"-Batch average results (epoch {epoch + 1 :.0f}) - Train obj: {c_bound/n_batches :.5f}, KL/n_batches: {c_kl/n_batches :.5f}, NLL loss: {c_loss/n_batches :.5f}, Train 0-1 Error:  {c_err/n_batches :.5f}")
    
    if learn_prior:
        return obj_progress, loss_progress, prior_sigma_progress
    else:
        return obj_progress, loss_progress

def testProbNN(net, eval_method, test_loader, n_samples, pmin, device):
    """
    Test performance of a probabilistic neural network via either stochastic, 
    posterior mean or ensemble prediction methods.

    Parameters
    ----------
    net : ProbNN object
        Probabilistic network to test.
    eval_method : string
        Evaluation method - 'stochastic', 'posterior-mean' or 'ensemble'.
    test_loader : DataLoader
        Test data.
    n_samples : int
        Only used if eval_method = 'ensemble', gives the number of 
        predictions to make in the ensemble.
    pmin : float
        Minimum probability used to upper bound CE loss.
    device : string
        Device to run code on.

    Returns
    -------
    loss : float
        Bounded cross entropy loss on the test set (ie averaged over batches).
    error : float
        01 error aka missclassification rate on the test set.

    """
    
    net.eval()
    correct, total, ce_loss = 0, 0, 0.0
    n_batches = len(test_loader)
    
    output = torch.zeros(test_loader.batch_size, net.n_classes).to(device)
    
    with torch.no_grad():
        
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            
            if eval_method == 'stochastic':
                # Each prediction requires a seperate draw of the weights
                for i in range(len(data)):
                    output[i, :] = net(data[i:i+1], pmin, sample=True)
                
            elif eval_method == 'posterior-mean':
                output = net(data, pmin, sample=False)
                
            elif eval_method == 'ensemble':
                 outputs = torch.zeros(n_samples, test_loader.batch_size, net.n_classes).to(device)
                 for i in range(n_samples):
                    outputs[i, :, :] = net(data, pmin, sample=True)
                 # Average over samples
                 output = outputs.mean(0)
            else: 
                print('Invalid evaluation method: {}'.format(eval_method))
                
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            # Work with bounded version of CE loss
            ce_loss += F.nll_loss(output, target) / np.log(1/pmin)
        
    loss = ce_loss.item()/n_batches
    error = 1 - (correct/total)
    
    return loss, error

# =============================================================================
# Evaluation Methods
# =============================================================================

def compute_losses(net, pmin, data = None, target = None, data_loader = None, mc_samples = None, delta_p = None, device = 'cpu'):
    """
    Computes losses, specific computations depends on input...
        - if a data_loader, mc_samples and delta_p are supplied, it will upper bound
        the average empirical risk with MC sampling and the Sample Convergence Bound
        - if mc_samples and delta_p are not both supplied it will simply compute
        the empirical (bounded) CE risk and the empirical (01) risk on the 
        data and target provided. These will be used later for computing
        gradients of the training objective.
    We also return the KL divergence of the network.

    Parameters
    ----------
    net : ProbNN object
        Network to compute losses on.
    pmin : float
        Minimum probability used to upper bound CE loss.
    data : single batch of a DataLoader, optional
        One batch of data to compute losses on. The default is None.
    target : single batch of a DataLoader, optional
        As above. The default is None.
    data_loader : DataLoader, optional
        Whole data in DataLoader object. The default is None.
    mc_samples : int, optional
        Number of samples in the MC routine. The default is None.
    """
    
    KL = net.compute_kl()
    
    if (mc_samples != None) and (data_loader != None) and (delta_p != None):
        KL = net.compute_kl().detach()
        mc_ce_loss, correct, total = 0.0, 0, 0
        n_batches = len(data_loader)
    
        output = torch.zeros(data_loader.batch_size, net.n_classes)
        
        with torch.no_grad():
            if n_batches == 1:
                for _, (data, target) in enumerate(data_loader):
                    data, target = data.to(device), target.to(device)
                    for m in tqdm(range(mc_samples), position=0, leave=True):
                        output = net(data, pmin, sample=True)
                        mc_ce_loss += float(F.nll_loss(output, target) / np.log(1/pmin))
                        pred = output.max(1, keepdim = True)[1]
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                ce_loss = mc_ce_loss/(n_batches*mc_samples)
                error = 1 - (correct/total)
            else:
                for _, (data, target) in enumerate(tqdm(data_loader)):
                    data, target = data.to(device), target.to(device)
                    for m in tqdm(range(mc_samples), position=0, leave=True):
                        output = net(data, pmin, sample=True)
                        mc_ce_loss += float(F.nll_loss(output, target) / np.log(1/pmin))
                        pred = output.max(1, keepdim = True)[1]
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                ce_loss = mc_ce_loss/(n_batches*mc_samples)
                error = 1 - (correct/total)
            
    
        # Bound the MC approximation of the empirical risk term
        # using the Sample Convergence Bound
        ce_loss_bound = inv_kl(ce_loss, np.log(2 / delta_p) / mc_samples)
        error_bound = inv_kl(error, np.log(2 / delta_p) / mc_samples)
        
        # These are the 'average emirical risks'
        return ce_loss_bound, error_bound, KL
    
    elif (data != None) and (target != None):
        KL = net.compute_kl()
        outputs = net(data, pmin, sample=True)
        if pmin != None:
            ce_loss = F.nll_loss(outputs, target) / np.log(1/pmin)
        else:
            ce_loss = F.nll_loss(outputs, target)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        error = 1-(correct/total)
        
        # These are the emirical risks (ie not averaged over Q), which 
        # can be used for computing gradients of the trianing objective.
        return ce_loss, error, KL
    
    else:
        print("Need to supply valid data combination, see docs")
        
    return None, None

def compute_bounds_and_metrics(net, bound_loader, bound, delta, delta_p, mc_samples, pmin, prior_weight, device):
    """
    Evaluate a PAC-Bayes upper bound on the generalisation error of the network.

    Parameters
    ----------
    net : ProbNN object
        The network to bound.
    bound_loader : DataLoader
        Data to evaluate bound, must be independent of data used to learn prior.
    bound : string
        Type of bound to compute. Current options 'inverse-kl', 'classic'.
    delta : float
        Probability threshold for PB bound ie. the bound will hold with probability
        at least 1-delta.
    delta_p : float
        Probability threshold for bounding the MC approximation of average risk,
        ie. the average empirical risk is less than equal the given quantity with
        probability at least 1-delta_p
    mc_samples : int
        Number of Monte Carlo samples used in upper bounding average empirical risk.
    pmin : float
        Minimum probability used to upper bound cross entropy loss.
    prior_weight : float
        prior weight which appears in the PB bound
    device : string
        Device to run on.

    Returns
    -------
    ce_bound : float
        PAC-Bayes upper bound on average risk using bounded CE loss.
    error_bound : float
        PAC-Bayes upper bound on average risk using 01 loss.
    train_obj : float
        Value of the training objective achieved on the bound data computed using MC approximation on the average risk.
    KL : float
        KL divergence between tfinal network and prior
    empirical_ce_risk_bound : float
        MC approximation to the average empirical CE risk on the bound data
    empirical_01_risk_bound : float
        MC approximation to the average empirical error on the bound data
    """
    
    net.eval()
    n = len(bound_loader.dataset)
    
    av_empirical_ce_risk_bound, av_empirical_01_risk_bound, KL = compute_losses(net, pmin, data_loader = bound_loader, mc_samples = mc_samples, delta_p = delta_p, device = device)
    
    ce_bound = compute_bound(av_empirical_ce_risk_bound, KL, n, delta, bound, prior_weight)
    error_bound = compute_bound(av_empirical_01_risk_bound, KL, n, delta, bound, prior_weight)
    
    return ce_bound, error_bound, KL, av_empirical_ce_risk_bound, av_empirical_01_risk_bound

def count_parameters(model): 
    """
    Counts the number of learnable parameters

    Parameters
    ----------
    model : ProbNN object
        The network to measure.

    Returns
    -------
    float
        Number of learnable parameters.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# Misc
# =============================================================================

def ResNet_extractSigma(net):
    """
    Pulls together all the variances for all the layers in a ResNet model since
    they are easily accessible in one place.

    Parameters
    ----------
    net : ProbNN object (ResNet)
        The network object to extract sigma from.

    Returns
    -------
    _sigma : array, float
        An array of values of sigma from the given network.

    """
    n_layers = net.depth
    _sigma = [0] * n_layers
    _sigma[0] = float(F.softplus(net.conv1.distributions.rho.detach()))
    inds = 2*np.array(net.blocks).cumsum()+1
    for i in range(net.blocks[0]):
        _sigma[1+2*i] = float(F.softplus(net.layer1[i].conv1.distributions.rho.detach()))
        _sigma[2+2*i] = float(F.softplus(net.layer1[i].conv2.distributions.rho.detach()))
    for i in range(net.blocks[1]):
        _sigma[inds[0]+2*i] = float(F.softplus(net.layer2[i].conv1.distributions.rho.detach()))
        _sigma[inds[0]+1+2*i] = float(F.softplus(net.layer2[i].conv2.distributions.rho.detach()))
    for i in range(net.blocks[2]):
        _sigma[inds[1]+2*i] = float(F.softplus(net.layer3[i].conv1.distributions.rho.detach()))
        _sigma[inds[1]+1+2*i] = float(F.softplus(net.layer3[i].conv2.distributions.rho.detach()))
    for i in range(net.blocks[3]):
        _sigma[inds[2]+2*i] = float(F.softplus(net.layer4[i].conv1.distributions.rho.detach()))
        _sigma[inds[2]+1+2*i] = float(F.softplus(net.layer4[i].conv2.distributions.rho.detach()))
    _sigma[-1] = float(F.softplus(net.fc.distributions.rho.detach()))
    return _sigma