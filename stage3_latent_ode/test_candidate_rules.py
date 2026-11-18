import unittest,numpy as np
from candidate_rules import fit_coefficients,relative_times,canonical,canonical_system_key
from types import SimpleNamespace
class Rules(unittest.TestCase):
 def test_time_origins(self):
  a=np.arange(40)*.005+1000.;b=np.arange(40)*.005+1.
  np.testing.assert_allclose(relative_times(a,2,16),relative_times(b,2,16),atol=1e-12)
 def test_scaled_fit(self):
  x=np.arange(1.,20);A=np.column_stack((x,1e6*x*x));c=fit_coefficients(A,2*x+3*x*x);np.testing.assert_allclose(c,[2,3e-6],rtol=1e-9)
 def test_collinear(self):
  with self.assertRaises(ValueError):fit_coefficients(np.ones((10,2)),np.ones(10))
 def test_nested_commutative_order_is_one_structure(self):
  first=SimpleNamespace(fields=('u','v'),rhs=(('add',('add',('v','u'),('v','v')),('n',1.0)),('v','u')))
  second=SimpleNamespace(fields=('u','v'),rhs=(('add',('n',1.0),('add',('v','v'),('v','u'))),('v','u')))
  self.assertEqual(canonical_system_key(first),canonical_system_key(second))
if __name__=='__main__':unittest.main()
