"""ODE counterpart of stage2 explicit syntax, identifiability and finite gates."""
import numpy as np
import itertools
import hashlib

def canonical_system_key(system):
 """Invariant under simultaneous state/equation permutations (up to six)."""
 fields=tuple(system.fields)
 if not 1<=len(fields)<=8:raise ValueError('Canonicalization supports 1..8 states')
 def rename(node,mapping):
  if node[0]=='v':return ('v',mapping[node[1]])
  children=[rename(x,mapping) if isinstance(x,tuple) else x for x in node[1:]]
  if node[0] in ('add','mul'):
   flattened=[]
   for child in children:
    flattened.extend(child[1:] if isinstance(child,tuple) and child[0]==node[0] else [child])
   children=sorted(flattened,key=repr)
  return (node[0],*children)
 signatures=[]
 # Eight-dimensional latent coordinates have fixed encoder-head ordering.
 # Deduplicate expressions in that order rather than enumerate 8! relabelings.
 permutations=itertools.permutations(range(len(fields))) if len(fields)<=6 else [tuple(range(len(fields)))]
 for permutation in permutations:
  mapping={f:permutation[i] for i,f in enumerate(fields)}
  equations=[None]*len(fields)
  for i,rhs in enumerate(system.rhs):equations[permutation[i]]=rename(rhs,mapping)
  signatures.append(repr(tuple(equations)))
 return hashlib.sha256(min(signatures).encode()).hexdigest()

def canonical(node):
 if not isinstance(node,tuple):return node
 children=[canonical(c) if isinstance(c,tuple) else c for c in node[1:]]
 if node[0] in ('add','mul'):children=sorted(children,key=repr)
 return (node[0],*children)

def validate_system(system):
 def check(n):
  if n[0] not in ('n','v','neg','add','mul','pow'):raise ValueError('Unsupported ODE operator: '+n[0])
  if n[0]=='pow' and (n[2][0]!='n' or not 0<=float(n[2][1])<=3):raise ValueError('Unsupported ODE exponent')
  for c in n[1:]:
   if isinstance(c,tuple):check(c)
 for rhs in system.rhs:check(rhs)
 return repr(tuple(canonical(r) for r in system.rhs))

def fit_coefficients(A,b):
 A=np.asarray(A,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if not np.isfinite(A).all() or not np.isfinite(b).all():raise ValueError('Nonfinite coefficient data')
 scale=np.linalg.norm(A,axis=0)
 if np.any(scale<1e-12):raise ValueError('Zero candidate column')
 normalized=A/scale
 if np.linalg.matrix_rank(normalized,tol=1e-7)<A.shape[1]:raise ValueError('Duplicate or collinear candidate columns')
 c=np.linalg.lstsq(normalized,b,rcond=1e-6)[0]/scale
 if not np.isfinite(c).all():raise ValueError('Nonfinite coefficients')
 return c

def relative_times(t,start,horizon):
 # Subtract in float64 before conversion; float32 absolute timestamps lose
 # precision differently at different origins, causing false interval mismatch.
 v=np.asarray(t,dtype=np.float64);r=v[start:start+horizon+1]-v[start]
 if len(r)!=horizon+1 or not np.all(np.diff(r)>0):raise ValueError('Invalid time grid')
 return r
