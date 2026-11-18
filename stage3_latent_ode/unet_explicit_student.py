"""Baseline U-Net encoder/decoder with an explicit spatial-channel ODE bottleneck.
All skip features originate in the visible input window. No teacher weights load.
"""
import torch
from torch import nn

class UNetExplicitStudent(nn.Module):
    architecture = 'unet_spatial_ode'
    def __init__(self, h, w, dim=2, width=64):
        super().__init__()
        from realpdebench.model.unet import Unet3d
        self.dim, self.h, self.w = dim, h, w
        self.net = Unet3d(dim=64, channels=3, out_channels=3,
                          dim_mults=(1,2,4), in_time=20, out_time=20)
        # Replace the original learned middle, retaining the exact down/up paths.
        for name in ('mid_block1','mid_spatial_attn','mid_temporal_attn','mid_block2'):
            setattr(self.net, name, nn.Identity())
        self.to_latent=nn.Conv2d(256,dim,1)
        self.from_latent=nn.Conv3d(dim,256,1)
        self.context=None

    def encoder(self, inputs):
        b=inputs.shape[0]
        x=inputs.reshape(b,20,2,self.h,self.w).permute(0,2,1,3,4)
        x=torch.cat((x,torch.zeros_like(x[:,:1])),1)
        n=self.net
        bias=n.time_rel_pos_bias(20,device=x.device)
        cond=n.time_mlp(torch.zeros(b,device=x.device))
        x=n.init_temporal_attn(n.init_conv(x),pos_bias=bias)
        initial=x
        skips=[]
        for block1,block2,spatial,temporal,down in n.downs:
            x=block2(block1(x,cond),cond)
            x=temporal(spatial(x),pos_bias=bias)
            skips.append(x)
            x=down(x)
        self.context=(initial,skips,cond,bias)
        return self.to_latent(x[:,:,-1]).permute(0,2,3,1)

    def forward(self, latent):
        if self.context is None:raise RuntimeError('Encode visible history before decoding')
        initial,skips,cond,bias=self.context
        b=initial.shape[0]
        if latent.shape[0]!=b*20:raise ValueError('Decoder expects 20 evolved states per visible history')
        x=latent.reshape(b,20,*latent.shape[1:]).permute(0,4,1,2,3)
        x=self.from_latent(x)
        for modules,skip in zip(self.net.ups,reversed(skips)):
            block1,block2,spatial,temporal,up=modules
            x=block2(block1(torch.cat((x,skip),1),cond),cond)
            x=up(temporal(spatial(x),pos_bias=bias))
        x=self.net.final_conv(torch.cat((x,initial),1))[:,:2]
        return x.permute(0,2,1,3,4).reshape(b*20,2,self.h,self.w)
