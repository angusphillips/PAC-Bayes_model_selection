#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:12:36 2021

@author: angusphillips
"""

import torch
import math
import numpy as np

def compute_bound(loss, KL, data_size, delta, bound, prior_weight):
    """ Computes the value of the named PAC-Bayes bound. May be 
    used as an evaluation or as a training objective, depending on how
    the provided loss was computed.
    
    Parameters
    ----------
    loss : float
        Loss
    data_size : int
        Size of data that the bound is evaluated on
    delta : float
        Threshold in the PB bound
    bound : string
        Type of bound to use. Options are 'classic', 'inverse-kl'
    """
    
    if bound == 'classic':
        return loss + torch.sqrt(torch.div((KL + np.log(1/prior_weight) + np.log(2*np.sqrt(data_size) / delta)),  2*data_size))
    
    if bound == 'inverse-kl':
        term2 =  (KL + np.log(1/prior_weight) + np.log(2 * np.sqrt(data_size) / delta)) / data_size
        return inv_kl(loss, term2)
    
    else:
        print('{} not a valid bound'.format(bound))
        return None
    
def inv_kl(qs, ks):
    # Implementation taken from https://github.com/mperezortiz/PBB
    """Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk
    ks : float
        second term for the binary kl inversion
    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1-1e-10
    while((dch-izq)/dch >= 1e-5):
        p = (izq+dch)*.5
        if qs == 0:
            ikl = ks-(0+(1-qs)*math.log((1-qs)/(1-p)))
        elif qs == 1:
            ikl = ks-(qs*math.log(qs/p)+0)
        else:
            ikl = ks-(qs*math.log(qs/p)+(1-qs) * math.log((1-qs)/(1-p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
        

