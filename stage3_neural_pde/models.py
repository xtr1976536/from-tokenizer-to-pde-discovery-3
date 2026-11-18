"""Spatial representations: no flattening, no future skip connection."""
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())

class Representation(nn.Module):
    def __init__(self,route,width=16,latent_stride=2):
        super().__init__()
        if route not in ('latent','physical') or latent_stride not in (1,2,4):
            raise ValueError('Unsupported route/stride')
        self.route=route;self.latent_stride=latent_stride
        self.shallow=Block(2,width);self.deep=Block(width,2*width)
        self.merge=Block(3*width,width);self.project=nn.Conv2d(width,2,1)
        # Two field-to-field channels make existing EqGPT grammar reusable.
        self.decoder=nn.Sequential(Block(2,width),nn.Conv2d(width,2,1))
    def encode(self,x):
        a=self.shallow(x);b=self.deep(F.avg_pool2d(a,2))
        h=self.merge(torch.cat((a,F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)),1))
        if self.route=='physical':return x+self.project(h)
        h=F.avg_pool2d(h,self.latent_stride) if self.latent_stride>1 else h
        return self.project(h)
    def decode(self,z,shape):
        # Decoder receives only evolved z; no observed or future encoder skips.
        z=F.interpolate(z,size=shape,mode='bilinear',align_corners=False)
        return self.decoder(z)
    def forward(self,x):
        z=self.encode(x)
        return self.decode(z,x.shape[-2:]) if self.route=='latent' else z
