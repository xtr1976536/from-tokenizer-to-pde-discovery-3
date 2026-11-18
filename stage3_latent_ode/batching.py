"""Bounded host/device memory with identical global-mean gradient semantics."""
import torch

def weighted_backward(batches,loss_function,total):
    value=None
    for batch in batches:
        weight=len(batch[0])/total
        loss=loss_function(batch)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite minibatch loss')
        (loss*weight).backward();term=loss.detach()*weight;value=term if value is None else value+term
    return float(value) if value is not None else 0.

def weighted_value(batches,loss_function,total):
    value=None
    with torch.no_grad():
        for batch in batches:
            term=loss_function(batch).detach()*len(batch[0])/total
            value=term if value is None else value+term
    return float(value) if value is not None else 0.
