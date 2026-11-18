"""Differentiable fixed-structure ODE core for alternating joint training.
No parameter detaches in prediction path; target encoding is stop-gradient.
Symbolic search remains outside autograd and must use frozen AE snapshots.
"""
import torch

def rk4(rhs,z,times,substeps=4):
    if times.ndim!=1 or len(times)<2 or substeps<1:raise ValueError('Invalid integration grid')
    # Validate once on CPU instead of synchronizing CUDA at every RK step.
    intervals=(times[1:]-times[:-1]).detach().cpu().tolist()
    if any(dt<=0 for dt in intervals):raise ValueError('Times must increase')
    states=[z]
    for i in range(len(times)-1):
        dt=intervals[i]/substeps
        for _ in range(substeps):
            k1=rhs(z);k2=rhs(z+dt*k1/2);k3=rhs(z+dt*k2/2);k4=rhs(z+dt*k3)
            z=z+dt*(k1+2*k2+2*k3+k4)/6
        states.append(z)
    result=torch.stack(states,dim=1)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite ODE rollout')
    return result

def objective(prediction,target,reconstruction,current,latent_prediction,latent_target,rec_weight=1.,latent_weight=.1):
    pred=(prediction-target).square().mean()
    rec=(reconstruction-current).square().mean()
    latent=(latent_prediction-latent_target.detach()).square().mean()
    return pred+rec_weight*rec+latent_weight*latent,dict(prediction=pred,reconstruction=rec,latent=latent)
