import torch
from torch import nn
from models import Block
class DistillStudent(nn.Module):
 def __init__(self,h,w,dim=8,width=8):
  super().__init__()
  self.encoder=nn.Sequential(Block(40,width),nn.AvgPool2d(2),Block(width,2*width),nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(64*width,dim))
  self.decoder=nn.Sequential(nn.Linear(dim,64*width),nn.Unflatten(1,(2*width,4,8)),nn.Upsample(size=(h,w),mode='bilinear',align_corners=False),Block(2*width,width),nn.Conv2d(width,2,1))
 def forward(self,z):return self.decoder(z)

class LatentRHS(nn.Module):
 def __init__(self,dim,width=32):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,dim))
 def forward(self,z):return self.net(z)

def distillation_loss(student, teacher, truth, lambda_teacher, lambda_real):
 if student.shape != truth.shape:
  raise ValueError('Student and real target rollout shapes must match')
 loss=lambda_real*(student-truth).square().mean()
 if lambda_teacher:
  if teacher is None or teacher.shape!=student.shape:
   raise ValueError('Teacher supervision requires matching teacher predictions')
  loss=loss+lambda_teacher*(student-teacher).square().mean()
 return loss

def make_student(h,w,dim,width,architecture='compact'):
 if architecture=='compact':return DistillStudent(h,w,dim,width)
 if architecture=='unet_spatial_ode':
  from unet_explicit_student import UNetExplicitStudent
  return UNetExplicitStudent(h,w,dim,width)
 raise ValueError('Unknown student architecture: '+architecture)
