"""Physical-coordinate numerical backend for scalar channels on a Cartesian grid.
Uses numpy.gradient with explicit coordinates and one-sided boundary stencils;
it never silently assumes periodicity. For noisy experimental data callers can
supply the same field/derivative callback contract using their surrogate model.
"""
from functools import lru_cache
import numpy as np
from ast_runtime import evaluate
class GridBackend:
    def __init__(self,coordinates,fields,parameters=None):
        self.coordinates={k:np.asarray(v,dtype=float) for k,v in coordinates.items()}
        self.axes=tuple(coordinates);shape=tuple(len(v) for v in self.coordinates.values())
        if not shape:raise ValueError('Coordinates required')
        for v in self.coordinates.values():
            if v.ndim!=1 or len(v)<3 or not np.isfinite(v).all() or np.any(np.diff(v)<=0):raise ValueError('Increasing finite coordinates (at least 3) required')
        self.values={}
        for name,a in fields.items():
            a=np.array(a,dtype=float,copy=True)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Field shape or values invalid')
            a.flags.writeable=False;self.values[name]=a
        for name,value in (parameters or {}).items():
            if name in self.values or name in self.coordinates:raise ValueError('Duplicate parameter/field name')
            self.values[name]=value
        self.values.update(dict(zip(self.axes,np.meshgrid(*self.coordinates.values(),indexing='ij'))))
        self.derivative=lru_cache(maxsize=2048)(self._derivative)
    def _derivative(self,node,axes):
        if any(a not in self.coordinates for a in axes):raise ValueError('Undeclared derivative coordinate')
        value=np.broadcast_to(evaluate(node,self.values,self.derivative),next(iter(self.values.values())).shape)
        for a in axes:value=np.gradient(value,self.coordinates[a],axis=self.axes.index(a),edge_order=2)
        return value
    def trajectory(self,fields,fit_mask,score_mask):
        if 't' not in self.coordinates:raise ValueError('Time coordinate required to compute targets')
        return dict(values=self.values,derivative=self.derivative,targets={f:self.derivative(('v',f),'t') for f in fields},fit_mask=fit_mask,score_mask=score_mask)
