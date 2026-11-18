import unittest,copy
import torch
from joint_loss import joint_objective,prepare_targets
from joint_core import rk4

class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*2*4*5,2));self.decoder=torch.nn.Sequential(torch.nn.Linear(2,2*4*5),torch.nn.Unflatten(1,(2,4,5)))
class Dyn(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64))
 def forward(self,z):return self.weight*z

def old_loss(m,dyn,b):
 x,y,t=b;z=m.encoder(x);zp=rk4(dyn,z,t,substeps=2);pred=m.decoder(zp.reshape(-1,2)).reshape_as(y)
 sequence=torch.cat((x.reshape(len(x),3,2,4,5),y[:,1:]),1)
 w=torch.stack([sequence[:,k:k+3].reshape(len(x),-1,4,5) for k in range(y.shape[1])],1)
 with torch.no_grad():target=m.encoder(w.reshape(-1,*x.shape[1:])).reshape(len(x),y.shape[1],2)
 return (pred[:,1:]-y[:,1:]).square().mean()+.5*(m.decoder(z)-y[:,0]).square().mean()+.1*(zp[:,1:]-target[:,1:]).square().mean()
class TestLoss(unittest.TestCase):
 def test_loss_gradients_and_optimizer_equivalence(self):
  torch.manual_seed(3);a=Model().double();b=copy.deepcopy(a);da=Dyn();db=copy.deepcopy(da)
  batch=(torch.randn(3,6,4,5,dtype=torch.float64),torch.randn(3,5,2,4,5,dtype=torch.float64),torch.arange(5,dtype=torch.float64)*.01)
  x=old_loss(a,da,batch);y=joint_objective(b,db,batch,3,2)
  torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12);x.backward();y.backward()
  for p,q in zip([*a.parameters(),*da.parameters()],[*b.parameters(),*db.parameters()]):torch.testing.assert_close(p.grad,q.grad,rtol=1e-11,atol=1e-12)
 def test_frozen_cached_coefficients_gradient(self):
  torch.manual_seed(5);m=Model().double();a=Dyn();b=copy.deepcopy(a)
  batch=(torch.randn(2,6,4,5,dtype=torch.float64),torch.randn(2,5,2,4,5,dtype=torch.float64),torch.arange(5,dtype=torch.float64)*.01)
  for p in m.parameters():p.requires_grad_(False)
  with torch.no_grad():prepared=(m.encoder(batch[0]),prepare_targets(m,batch,3,2))
  x=joint_objective(m,a,batch,3,2);y=joint_objective(m,b,batch,3,2,prepared)
  x.backward();y.backward();torch.testing.assert_close(x,y);torch.testing.assert_close(a.weight.grad,b.weight.grad,rtol=1e-12,atol=1e-12)
if __name__=='__main__':unittest.main()
