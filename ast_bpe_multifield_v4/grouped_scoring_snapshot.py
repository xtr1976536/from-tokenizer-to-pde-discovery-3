"""Unified group-balanced parametric structure score.
Input groups=[{parameters:{...}, trajectories:[...]}]. Same-parameter groups
must be merged by callers; flat legacy trajectory input is one constant group.
"""
import numpy as np

def normalize(data,basis):
 if isinstance(data,dict) and 'groups' in data:groups=data['groups']
 elif isinstance(data,dict) and 'trajectories' in data:groups=[data]
 else:groups=[dict(parameters={},trajectories=[data] if isinstance(data,dict) else data)]
 if not groups:raise ValueError('No parameter groups')
 merged={}
 for g in groups:
  params=g.get('parameters',{});key=tuple(sorted(params.items()))
  merged.setdefault(key,dict(parameters=params,trajectories=[]))['trajectories'].extend(g['trajectories'])
 groups=list(merged.values())
 if any(not g['trajectories'] for g in groups):raise ValueError('Empty group')
 B=np.asarray([basis_values(g['parameters'],basis) for g in groups])
 if np.linalg.matrix_rank(B)<B.shape[1]:raise ValueError('Parameter basis not identifiable across groups')
 return groups,B

def basis_values(parameters,basis):
 values=[]
 for name in basis:
  if name=='1':v=1.
  elif name.startswith('1/'):
   p=float(parameters[name[2:]])
   if p==0:raise ValueError('Zero inverse parameter')
   v=1/p
  else:v=float(parameters[name])
  if not np.isfinite(v):raise ValueError('Nonfinite parameter basis')
  values.append(v)
 return values

def fit(system,data,basis,design,solve,epi=.2,export=False):
 groups,B=normalize(data,basis);scores=[];coefficients={};models={}
 for field,rhs in zip(system.fields,system.rhs):
  blocks=[];matrices=[];targets=[]
  for gi,g in enumerate(groups):
   for tr in g['trajectories']:
    X,y=design(rhs,tr,field);mask=np.asarray(tr['fit_mask'],bool).ravel()
    if mask.size!=y.size or not mask.any():raise ValueError('Invalid fit mask')
    # basis-major columns: [X, parameter1*X, ...]
    expanded=np.hstack([b*X for b in B[gi]])
    scale=np.sqrt(len(groups)*len(g['trajectories'])*mask.sum()*max(np.mean(y[mask]**2),1e-24))
    matrices.append(expanded[mask]/scale);targets.append(y[mask]/scale);blocks.append((gi,X,y,mask))
  beta=solve(matrices,targets).reshape(len(basis),-1);group_scores=[[] for _ in groups]
  for gi,X,y,mask in blocks:
   coef=B[gi]@beta;denom=np.sum((y[mask]-y[mask].mean())**2)
   if denom<1e-24:raise ValueError('R2 undefined for constant target')
   group_scores[gi].append(1-float(np.sum((y[mask]-X[mask]@coef)**2)/denom))
  r2=float(np.mean([np.mean(v) for v in group_scores]));count=beta.shape[1]+1
  scores.append(dict(field=field,r2=r2,term_columns=count,reward=(1-epi*np.log10(count))*r2,group_r2=[float(np.mean(v)) for v in group_scores]))
  models[field]=beta.tolist()
  if len(basis)==1 and basis[0]=='1':coefficients[field]=beta[0].tolist()
 result=dict(status='evaluated',reward=float(np.mean([s['reward'] for s in scores])),fields=scores,groups=len(groups),trajectories=sum(len(g['trajectories']) for g in groups),parameter_basis=list(basis),weighting='equal groups, equal trajectories within group, target RMS and point count normalized',coefficient_policy='temporary joint OLS for structure scoring; final parameters fitted separately')
 if export:result.update(method='post-search group-balanced parametric least squares',coefficient_models=models,coefficients=coefficients)
 return result

def coefficients_at(final,parameters):
 b=np.asarray(basis_values(parameters,final['parameter_basis']))
 return {f:(b@np.asarray(beta)).tolist() for f,beta in final['coefficient_models'].items()}
