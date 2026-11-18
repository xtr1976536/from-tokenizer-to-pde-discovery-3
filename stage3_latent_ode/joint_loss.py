"""Reusable joint objective with exact frozen-representation caching."""
import torch
from joint_core import rk4

def prepare_targets(model,b,history,dim):
    x,y,t=b
    sequence=torch.cat((x.reshape(len(x),history,2,*y.shape[-2:]),y[:,1:]),dim=1)
    windows=sequence.unfold(1,history,1).permute(0,1,5,2,3,4)
    windows=windows.reshape(-1,history*2,*y.shape[-2:])
    if x.is_contiguous(memory_format=torch.channels_last):windows=windows.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():return model.encoder(windows).reshape(len(x),y.shape[1],dim)

def joint_objective(model,dyn,b,history,dim,prepared=None):
    x,y,t=b
    if prepared is None:
        z=model.encoder(x);target=prepare_targets(model,b,history,dim)
    else:z,target=prepared
    latent=rk4(dyn,z,t,substeps=2)
    # The old implementation decoded z twice: once in rollout, once for rec.
    # Encode/decode network has no dropout or batchnorm, so this is identical.
    physical=model.decoder(latent.reshape(-1,dim)).reshape_as(y)
    return ((physical[:,1:]-y[:,1:]).square().mean()
            +.5*(physical[:,0]-y[:,0]).square().mean()
            +.1*(latent[:,1:]-target[:,1:]).square().mean())
