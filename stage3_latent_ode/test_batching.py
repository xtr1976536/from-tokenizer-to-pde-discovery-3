import unittest,copy
import torch
from batching import weighted_backward,weighted_value
class TestBatching(unittest.TestCase):
 def test_uneven_chunks_preserve_loss_gradient_adam_update(self):
  torch.manual_seed(9);a=torch.nn.Linear(3,2).double();b=copy.deepcopy(a);x=torch.randn(11,3,dtype=torch.float64);y=torch.randn(11,2,dtype=torch.float64)
  opta=torch.optim.Adam(a.parameters(),lr=1e-4);optb=torch.optim.Adam(b.parameters(),lr=1e-4)
  loss=(a(x)-y).square().mean();loss.backward()
  chunks=[(x[i:i+4],y[i:i+4]) for i in range(0,11,4)]
  val=weighted_backward(iter(chunks),lambda pair:(b(pair[0])-pair[1]).square().mean(),11)
  self.assertAlmostEqual(loss.item(),val,places=12)
  for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-12,atol=1e-12)
  opta.step();optb.step()
  for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=1e-12,atol=1e-12)
if __name__=='__main__':unittest.main()
