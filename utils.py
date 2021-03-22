"""
    Utility module for TIMES

    @author: Younghyun Kim
    @edited: 2020.04.03.
"""

import sys
import math
import time
import os
import shutil
import torch
import torch.distributions as dist
import numpy as np

# Functions
def save_vars(vs, filepath):
    """
        Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)

def save_model(model, filepath):
    """
        To load a saved model, simply use
        'model.load_state_dict(torch.load('path-to-saved-model'))'.
    """
    save_vars(model.state_dict(), filepath)

def log_mean_exp(value, dim=0, keepdim=False):
    " log mean exp "
    return torch.logsumexp(value, dim, keepdim=keepdim) -\
            math.log(value.size(dim))

def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY

def kl_divergence(p, q, samples=None):
    if has_analytic_kl(type(p), type(q)):
        return dist.kl_divergence(p, q)
    else:
        if samples is None:
            K = 10
            samples = p.rsample(torch.Size([K])) \
                    if p.has_rsample else p.sample(torch.Size([K]))
        ent = -p.log_prob(samples)
        return (-ent - q.log_prob(samples)).mean(0)
