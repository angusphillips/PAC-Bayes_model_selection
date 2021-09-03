#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:37:47 2021

@author: angusphillips

This module implements the required architectures for the dissertation. Each
architecture has a standard and probabilitic implementation. In standard implementations
parameters are initialised by one of the build in torch.nn modules. In probabilistic
implementations, parameters are stored as mean and variance of a gaussian variational
distribution using the Gaussian_VI class.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FCN architectures
# =============================================================================
class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.type = 0
        
        sigma_weights = np.sqrt(2/n_in)
        
        # Initialse according to truncated normal
        self.weight = nn.Parameter(trunc_normal_(torch.Tensor(n_out, n_in), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_out), requires_grad=True)
            
    def forward(self, input):
        # Forward pass
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)

class FCN(nn.Module):
    def __init__(self, image_size, n_classes, width, depth, dropout_prob):
        super().__init__()
        if not isinstance(width, list):
            width = np.ones(depth, dtype = int) * width
        self.width = width
        self.depth = depth
        self.image_size = image_size
        self.n_features = int(image_size[1]*image_size[2])
        self.n_classes = n_classes
        layers = [0] * (self.depth+1)
        if self.depth == 0:
            layers[0] = LinearLayer(self.n_features, self.n_classes)
        else:
            layers[0] = LinearLayer(self.n_features, self.width[0])
            layers[self.depth] = LinearLayer(self.width[-1], self.n_classes)
            for i in range(self.depth-1):
                layers[i+1] = LinearLayer(self.width[i], self.width[i+1])
        self.layers = nn.ModuleList(layers)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x, pmin):
        if self.depth == 0:
            x = x.view(-1, self.n_features)
            x = self.d(self.layers[0](x))
            x = output_transform(x, pmin)
        else:
            x = x.view(-1, self.n_features)
            for i in range(self.depth):
                x = F.relu(self.d(self.layers[i](x)))
            x = self.layers[self.depth](x)
            x = output_transform(x, pmin)
        return x

class ProbLinearLayer(nn.Module):  
    def __init__(self, n_in, n_out, rho_prior, learn_prior, init_layer, device):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rho_prior = rho_prior
        self.learn_prior = learn_prior
        self.type = 1
        self.device = device
        
        sigma_weights = np.sqrt(2/n_in)
        
        # Initialise means according to the learnt prior
        if init_layer != None:
            if init_layer.type == 0:
                self.mu_w_prior = init_layer.weight.detach()
                self.mu_b_prior = init_layer.bias.detach()
            elif init_layer.type == 1:
                self.mu_w_prior = init_layer.distributions.mu_w.clone()
                self.mu_b_prior = init_layer.distributions.mu_b.clone()
                self.rho_prior = float(init_layer.distributions.rho.detach())
        else:
            # Initialise distribution means using truncated normal
            self.mu_w_prior = trunc_normal_(torch.Tensor(n_out, n_in), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            self.mu_b_prior = torch.zeros(n_out)
            
        self.distributions = Gaussian_VI(self.learn_prior, self.rho_prior, self.mu_w_prior, self.mu_b_prior, self.device)
            
    # Forward pass - is required to compute output as well as KL
    def forward(self, input, sample = False):
        if self.training or sample:
            # Random sample of parameters
            weight = self.distributions.sample_weights()
            bias = self.distributions.sample_bias()
        else:
            # Current mean of parameters
            weight = self.distributions.mu_w
            bias = self.distributions.mu_b

        return F.linear(input, weight, bias)

class ProbFCN(nn.Module):
    def __init__(self, image_size, n_classes, width, depth, sigma_prior, learn_prior, init_net, device):
        super().__init__()
        
        # Model attributes
        self.image_size = image_size
        self.n_features = image_size[1]*image_size[2]
        self.n_classes = n_classes
        if not isinstance(width, list):
            width = np.ones(depth, dtype = int) * width
        self.width = width
        self.depth = depth
        self.learn_prior = learn_prior
        self.device = device
        
        rho_prior = torch.tensor(math.log(math.exp(sigma_prior)-1.0))
        
        # Set up each layer using the layers of init_net as priors
        layers = [0] * (self.depth+1)
        if self.depth == 0:
            layers[0] = ProbLinearLayer(self.n_features, n_classes, rho_prior, learn_prior, init_layer = init_net.layers[0] if (init_net != None) else None, device = self.device)
        else: 
            layers[0] = ProbLinearLayer(self.n_features, width[0], rho_prior, learn_prior, init_layer = init_net.layers[0] if (init_net != None) else None, device = device)
            layers[self.depth] = ProbLinearLayer(self.width[-1], n_classes, rho_prior, learn_prior, init_layer = init_net.layers[self.depth] if (init_net != None) else None, device = self.device)
            for i in range(self.depth-1):
                layers[i+1] = ProbLinearLayer(self.width[i], self.width[i+1], rho_prior, learn_prior, init_layer = init_net.layers[i+1] if (init_net != None) else None, device = self.device)
        
        # Put layers into container
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, pmin, sample=False):
        if self.depth == 0:
            x = x.view(-1, self.n_features)
            x = self.layers[0](x, sample)
            x = output_transform(x, pmin)
        else:
            x = x.view(-1, self.n_features)
            for i in range(self.depth):
                x = F.relu(self.layers[i](x, sample))
            x = self.layers[self.depth](x, sample)
            x = output_transform(x, pmin)
        return x
    
    def compute_kl(self):
        kl = torch.zeros(1).to(self.device)
        for i in range(self.depth+1):
            kl = kl + self.layers[i].distributions.compute_kl()
        return kl
    
    
# =============================================================================
# Convolutional architectures
# =============================================================================
     
class CNN(nn.Module):
    def __init__(self, image_size, n_classes, width, depth, kernel_size, padding, device):
        super().__init__()
        self.image_size = image_size
        self.n_classes = n_classes
        self.depth = depth
        self.width = [width * (2 ** i) for i in range(depth)]
        self.kernel_size = kernel_size
        self.padding = padding
        self.device = device
        
        # Convolutional layers
        layers = [0] * (self.depth+1)
        layers[0] = nn.Conv2d(in_channels=self.image_size[0], out_channels=self.width[0],
                                   kernel_size=self.kernel_size, stride=1, padding=self.padding, device=self.device)
        for i in range(self.depth-1):
            layers[i+1] = nn.Conv2d(in_channels=self.width[i], out_channels=self.width[i+1],
                                         kernel_size=self.kernel_size, stride=1, padding=self.padding, device=self.device)
            
        # Batch Normalisations
        batchnorms = [0]*self.depth
        for i in range(self.depth):
            batchnorms[i] = nn.BatchNorm2d(self.width[i])
        self.bn = nn.ModuleList(batchnorms)
        
        # Fully-connected layer
        # This works for the specific architecture only
        self.n_features = int(self.width[-1]*self.image_size[1]*self.image_size[2]/(4**(self.depth-1)))
        
        layers[self.depth]= LinearLayer(self.n_features, self.n_classes)
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, pmin):
        x = F.relu(self.bn[0](self.layers[0](x)))
        for i in range(self.depth-1):
            x = F.relu(self.bn[i+1](self.layers[i+1](x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = output_transform(self.layers[self.depth](x), pmin)
        return x
    

class ProbConvLayer(nn.Module): 
    def __init__(self, in_channels, out_channels, rho_prior, kernel_size, stride, padding, learn_prior, init_layer, device):
        super().__init__()
        self.prob = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.learn_prior = learn_prior
        self.rho_prior = rho_prior
        self.device = device
        
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = np.sqrt(2/in_features)
        
        # Initialise means according to the learnt prior
        if init_layer != None:
            if hasattr(init_layer, 'prob'):
                self.mu_w_prior = init_layer.distributions.mu_w.clone()
                self.mu_b_prior = init_layer.distributions.mu_b.clone()
                self.rho_prior = float(init_layer.distributions.rho.detach())
            else:
                self.mu_w_prior = init_layer.weight.detach()
                self.mu_b_prior = init_layer.bias.detach()
        else:
            # Initialise distribution means using truncated normal
            self.mu_w_prior = trunc_normal_(torch.Tensor(out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            self.mu_b_prior = torch.zeros(out_channels)
            
        self.distributions = Gaussian_VI(self.learn_prior, self.rho_prior, self.mu_w_prior, self.mu_b_prior, self.device)
            
    # Forward pass - is required to compute output as well as KL
    def forward(self, input, sample = False):
        if self.training or sample:
            # Random sample of parameters
            weight = self.distributions.sample_weights()
            bias = self.distributions.sample_bias()
        else:
            # Current mean of parameters
            weight = self.distributions.mu_w
            bias = self.distributions.mu_b
            
        return F.conv2d(input, weight, bias, stride=self.stride, padding=self.padding)

class ProbCNN(nn.Module):
    def __init__(self, image_size, n_classes, width, depth, kernel_size, padding, sigma_prior, learn_prior, init_net, device):
        super().__init__()
        self.image_size = image_size
        self.n_classes = n_classes
        self.learn_prior = learn_prior
        self.depth = depth
        self.width = [width * (2 ** i) for i in range(depth)]
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = padding
        self.device = device
        rho_prior = torch.tensor(math.log(math.exp(sigma_prior)-1.0))
        
        # Convolutional layers
        layers = [0] * (self.depth+1)
        layers[0] = ProbConvLayer(self.image_size[0], self.width[0], rho_prior, self.kernel_size, self.stride, self.padding, self.learn_prior, init_net.layers[0] if (init_net != None) else None, self.device)
        for i in range(self.depth-1):
            layers[i+1] = ProbConvLayer(self.width[i], self.width[i+1], rho_prior, self.kernel_size, self.stride, self.padding, self.learn_prior, init_net.layers[i+1] if (init_net != None) else None, self.device)
        
        # BatchNorm layers
        bn = [0] * self.depth
        for i in range(self.depth):
            bn[i] = nn.BatchNorm2d(self.width[i])
        self.bn = nn.ModuleList(bn)
        
        # Dense layer
        self.n_features = int(self.width[-1]*self.image_size[1]*self.image_size[2]/(4**(self.depth-1)))
        layers[self.depth] = ProbLinearLayer(self.n_features, n_classes, rho_prior, self.learn_prior, init_net.layers[self.depth] if (init_net != None) else None, self.device)
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, pmin, sample = False):
        x = F.relu(self.bn[0](self.layers[0](x, sample)))
        for i in range(self.depth-1):
            x = F.relu(self.bn[i+1](self.layers[i+1](x, sample)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.layers[self.depth](x, sample)
        x = output_transform(x, pmin)
        return x

    def compute_kl(self):
        kl = torch.zeros(1).to(self.device)
        for i in range(self.depth+1):
            kl = kl + self.layers[i].distributions.compute_kl()
        return kl

# =============================================================================
# ResNet architectures    
# =============================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, downsample):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=3, 
                               stride=2 if downsample else 1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3,  
                              stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.channels)
        
        if downsample:
            self.conv_ds = nn.Conv2d(self.in_channels, self.channels, kernel_size=1,
                                     stride=2, padding=0)
            self.bn_ds = nn.BatchNorm2d(self.channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample:
            identity = self.bn_ds(self.conv_ds(identity))
            x = x + identity
        else:
            x = x + identity
        x = F.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, image_size, n_classes, width, blocks):
        super().__init__()
        self.image_size = image_size
        self.n_classes = n_classes
        self.blocks = blocks
        self.depth = sum(self.blocks)*2 + 2
        self.width = width
        self.RN = True
        
        self.conv1 = nn.Conv2d(image_size[0], width, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self.make_layer(width, width, blocks[0])
        self.layer2 = self.make_layer(width, width * 2, blocks[1])
        self.layer3 = self.make_layer(width * 2, width * 4, blocks[2])
        self.layer4 = self.make_layer(width * 4, width * 8, blocks[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = width * 8
        self.fc = LinearLayer(in_features, self.n_classes)
        
    def make_layer(self, in_channels, channels, num_blocks):
        layers = [0] * num_blocks
        layers[0] = BasicBlock(in_channels, channels, downsample=not(in_channels==channels))
        for i in range(num_blocks-1):
            layers[i+1] = BasicBlock(channels, channels, downsample=False)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = output_transform(x)
        return x
    
class ProbBasicBlock(nn.Module):
    def __init__(self, in_channels, channels, rho_prior, learn_prior, init_block, downsample, device):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.downsample = downsample
        self.learn_prior = learn_prior
        self.device = device
        
        self.conv1 = ProbConvLayer(self.in_channels, self.channels, rho_prior, kernel_size=3, stride=2 if downsample else 1, padding=1, learn_prior=self.learn_prior, init_layer=init_block.conv1 if (init_block != None) else None, device = self.device)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.conv2 = ProbConvLayer(self.channels, self.channels, rho_prior, kernel_size=3, stride=1, padding=1, learn_prior=self.learn_prior, init_layer=init_block.conv2 if  (init_block != None) else None, device = self.device)
        self.bn2 = nn.BatchNorm2d(self.channels)
        
        if downsample:
            self.conv_ds = nn.Conv2d(self.in_channels, self.channels, kernel_size=1, 
                                     stride=2, padding=0)
            self.bn_ds = nn.BatchNorm2d(self.channels)

    def forward(self, x, sample):
        identity = x
        x = F.relu(self.bn1(self.conv1(x, sample)))
        x = self.bn2(self.conv2(x, sample))
        if self.downsample:
            identity = self.bn_ds(self.conv_ds(identity))
            x = x + identity
        else:
            x = x + identity
        x = F.relu(x)
        return x
    
    def compute_block_kl(self):
        kl = torch.zeros(1).to(self.device)
        kl = kl + self.conv1.distributions.compute_kl() + self.conv2.distributions.compute_kl()
        return kl
        

class ProbResNet(nn.Module):
    def __init__(self, image_size, n_classes, width, blocks, sigma_prior, learn_prior, init_net, device):
        super().__init__()
        self.image_size = image_size
        self.n_classes = n_classes
        self.blocks = blocks
        self.depth = sum(self.blocks)*2 + 2
        self.width = width
        self.learn_prior = learn_prior
        self.rho_prior = torch.tensor(math.log(math.exp(sigma_prior)-1.0))
        self.init_net = init_net
        self.device = device
        
        self.conv1 = ProbConvLayer(image_size[0], width, self.rho_prior, kernel_size=3, stride=1, padding=1, learn_prior=self.learn_prior, init_layer = self.init_net.conv1 if (init_net != None) else None, device=self.device)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self.make_layer(width, width, blocks[0], self.learn_prior, self.init_net.layer1)
        self.layer2 = self.make_layer(width, width * 2, blocks[1], self.learn_prior, self.init_net.layer2)
        self.layer3 = self.make_layer(width * 2, width * 4, blocks[2], self.learn_prior, self.init_net.layer3)
        self.layer4 = self.make_layer(width * 4, width * 8, blocks[3], self.learn_prior, self.init_net.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = width * 8
        self.fc = ProbLinearLayer(in_features, self.n_classes, self.rho_prior, self.learn_prior, self.init_net.fc, self.device)
        
    def make_layer(self, in_channels, channels, num_blocks, learn_prior, init_layer):
        layers = [0] * num_blocks
        layers[0] = ProbBasicBlock(in_channels, channels, self.rho_prior, learn_prior, init_layer[0], downsample=not(in_channels==channels), device=self.device)
        for i in range(num_blocks-1):
            layers[i+1] = ProbBasicBlock(channels, channels, self.rho_prior, learn_prior, init_layer[i+1], downsample=False, device = self.device)
        return nn.ModuleList(layers)
    
    def forward(self, x, pmin, sample):
        x = F.relu(self.bn1(self.conv1(x, sample)))
                
        for i in range(self.blocks[0]):
            x = self.layer1[i](x, sample)
        for i in range(self.blocks[1]):
            x = self.layer2[i](x, sample)
        for i in range(self.blocks[2]):
            x = self.layer3[i](x, sample)
        for i in range(self.blocks[3]):
            x = self.layer4[i](x, sample)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, sample)
        
        x = output_transform(x, pmin)
        return x
    
    def compute_kl(self):
        kl = torch.zeros(1).to(self.device)
        kl = kl + self.conv1.distributions.compute_kl()
        for i in range(self.blocks[0]):
            kl = kl + self.layer1[i].compute_block_kl()
        for i in range(self.blocks[1]):
            kl = kl + self.layer2[i].compute_block_kl()
        for i in range(self.blocks[2]):
            kl = kl + self.layer3[i].compute_block_kl()
        for i in range(self.blocks[3]):
            kl = kl + self.layer4[i].compute_block_kl()
        kl = kl + self.fc.distributions.compute_kl()
            
        return kl

# =============================================================================
# Utilities
# =============================================================================

class Gaussian_VI(nn.Module):
    """
    Deals with the learnable parameters in the probabilistic interpretations.
    Parameters are interpreted as multivariate Gaussians, so we store means 
    and isotropic variance matrix which are initialised according to the 
    values passed. If the learn_prior flag is true, we use the 
    adapted structure (i.e. shared variance between prior and posterior) as explained
    in the dissertation.
    
    Parameters
    ----------
    learn_prior : bool
        Whether the model is being used in stage 2 or stage 3 (see dissertation)
    rho_init : float
        The initial value of parameter rho, which is transformed into the posterior
        variance using softplus
    mu_w_prior : array, float
        Prior means for weight parameters
    mu_b_prior : array, float
        Prior means for bias parameters
    device : 'cpu' or 'cuda'
        Device to perform computations on.    
    
    """
    def __init__(self, learn_prior, rho_init, mu_w_prior, mu_b_prior, device):
        super().__init__()
        self.learn_prior = learn_prior
        self.mu_w_prior = mu_w_prior
        self.mu_b_prior = mu_b_prior
        self.mu_w = nn.Parameter(self.mu_w_prior.clone(), requires_grad = True)
        self.mu_b = nn.Parameter(self.mu_b_prior.clone(), requires_grad = True)
        if self.learn_prior:
            self.rho = nn.Parameter(torch.ones(1) * rho_init, requires_grad = True)
        else:
            self.rho = nn.Parameter(torch.ones(1) * rho_init, requires_grad = False)
            self.rho_w_ = nn.Parameter(torch.ones_like(self.mu_w_prior) * rho_init, requires_grad = True)
            self.rho_b_ = nn.Parameter(torch.ones_like(self.mu_b_prior) * rho_init, requires_grad = True)
        self.device = device
    
    @property
    def rho_w_prior(self):
        return torch.ones_like(self.mu_w_prior) * self.rho
    
    @property
    def rho_b_prior(self):
        return torch.ones_like(self.mu_b_prior) * self.rho
    
    @property
    def rho_w(self):
        if self.learn_prior:
            return torch.ones_like(self.mu_w_prior) * self.rho
        else:
            return self.rho_w_
    
    @property
    def rho_b(self):
        if self.learn_prior:
            return torch.ones_like(self.mu_b_prior) * self.rho
        else:
            return self.rho_b_
    
    def sample_weights(self):
        epsilon = torch.randn(self.mu_w.size()).to(self.device)
        return self.mu_w + epsilon * F.softplus(self.rho_w)
    
    def sample_bias(self):
        epsilon = torch.randn(self.mu_b.size()).to(self.device)
        return self.mu_b + epsilon * F.softplus(self.rho_b)
         
    def compute_kl(self):
        # Compute KL divergence between posterior and prior
        var0_w = torch.pow(F.softplus(self.rho_w_prior), 2)
        var1_w = torch.pow(F.softplus(self.rho_w), 2)
        term1_w = torch.log(torch.div(var0_w, var1_w))
        term2_w = torch.div(torch.pow(self.mu_w - self.mu_w_prior, 2), var0_w)
        term3_w = torch.div(var1_w, var0_w)
        kl_div_w = (torch.mul(term1_w + term2_w + term3_w - 1, 0.5)).sum()
        var0_b = torch.pow(F.softplus(self.rho_b_prior), 2)
        var1_b = torch.pow(F.softplus(self.rho_b), 2)
        term1_b = torch.log(torch.div(var0_b, var1_b))
        term2_b = torch.div(torch.pow(self.mu_b - self.mu_b_prior, 2), var0_b)
        term3_b = torch.div(var1_b, var0_b)
        kl_div_b = (torch.mul(term1_b + term2_b + term3_b - 1, 0.5)).sum()
        kl = kl_div_w + kl_div_b
        return kl

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
        
    Parameters
    ----------
    tensor : torch.tensor
        Tensor to fill
    mean : float
        Distribution mean
    std : float
        Distribution standard deviation
    a : float
        Lower truncation point
    b : float
        Upper truncation point
        
    Returns
    -------
    tensor : torch.tensor
        Original tensor filled with draws from the truncated normal
    """
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std)
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
        
def output_transform(x, pmin=None):
    """
    Computes the log softmax and clamps the values using the
    min probability given by pmin.

    Parameters
    ----------
    x : tensor
        output of the network
    pmin : float
        threshold of probabilities to clamp
    clamping : bool
        whether to clamp the output probabilities
    
    Returns
    -------
    output: torch.tenso
        Output of the network in the form of log-softmax probabilities
        for each class and each input
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if pmin != None:
        output = torch.clamp(output, min = np.log(pmin))
    return output