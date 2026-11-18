"""Explicit two-field evolution task, preserving each Laplacian as one term."""
import sys
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'ast_bpe_multifield_v1'))
sys.path.insert(0,str(ROOT.parent/'ast_bpe_multifield_v2'))
from coefficient_free import Parser
from ast_runtime import terms,expand_laplacian,evaluate
from grouped_scoring_snapshot import fit
sys.path.insert(0,str(ROOT))

@dataclass
class System:
    fields:tuple
    rhs:tuple
    text:str
    spatial:tuple=('y','x')


def validate_generated_text(text,fields=('u','v')):
    """Strict textual gate before parsing model-generated LaTeX."""
    if not isinstance(text,str) or not text.strip():raise ValueError('Empty generated formula')
    if len(text.split(r'\\'))!=len(fields):raise ValueError('Wrong number of generated equations')
    for equation in text.split(r'\\'):
        if re.search(r'=\s*\+',equation):raise ValueError('Redundant leading plus after equality')
        if re.search(r'(?<![eE])(?:\+\+|--|\+-|-\+)',equation):raise ValueError('Consecutive arithmetic signs')
        if re.search(r'\\partial(?:_\{[txy]+\}|_[txy]+)\s*(?!\()',equation):raise ValueError('Derivative operand must be parenthesized')
        if re.search(r'\\Delta\s*(?!\()',equation):raise ValueError('Laplacian operand must be parenthesized')
        if re.search(r'\)\s*(?!\\cdot\b)(?=[A-Za-z{(\\])',equation):raise ValueError('Implicit multiplication after group')
        if re.search(r'\}\s*(?!\\cdot\b)(?=[A-Za-z\\])',equation):raise ValueError('Implicit multiplication after superscript/group')
        if re.search(r'[uvp0-9]\s*(?=(?:[uvp]|\\partial|\\Delta|\\nabla|\())',equation):raise ValueError('Implicit multiplication requires \\cdot')
        if re.search(r'\)\s*\{',equation):raise ValueError('Stray brace after group')
        if re.search(r'\}\s*\{',equation):raise ValueError('Adjacent isolated braces')
    return True

def temporal(node):
    return node[0]=='d' and 't' in node[2] or any(temporal(v) for v in node[1:] if isinstance(v,tuple))

def derivative_order(node):
    order=len(node[2]) if node[0]=='d' else 2*node[2] if node[0]=='lap' else 1 if node[0] in ('grad','divergence') else 0
    nested=max([derivative_order(v) for v in node[1:] if isinstance(v,tuple)] or [0])
    return order+nested if node[0] in ('d','lap','grad','divergence') else nested

def parse_system(text,fields=('u','v'),spatial=('y','x')):
    pieces=text.split(r'\\')
    if len(pieces)!=len(fields):raise ValueError('Wrong number of task fields')
    rhs=[]
    for field,piece in zip(fields,pieces):
        if piece.count('=')!=1:raise ValueError('Explicit equality required')
        left,right=piece.split('=')
        args=dict(fields=fields,coordinates=('t',)+spatial)
        lhs=Parser(left,**args).parse();node=Parser(right,**args).parse()
        target=('d',('v',field),'t')
        parts=terms(lhs)
        if parts.count(target)!=1:raise ValueError('First-order unit evolution target required')
        remaining=list(parts);remaining.remove(target)
        node=('add',node,*(('neg',v) for v in remaining)) if remaining else node
        if temporal(node):raise ValueError('Implicit time derivative cannot enter this pilot solver')
        max_order=3 if __import__('os').environ.get('AST_BPE_BROAD_STRUCTURES')=='1' else 2
        if derivative_order(node)>max_order:raise ValueError('Spatial derivative order exceeds active structure protocol')
        # Bound numerical task scope before scoring; this does not constrain LM sampling.
        def check(n):
            if n[0] not in ('n','v','neg','add','mul','div','pow','d','lap','grad','divergence','fn'):raise ValueError('Unsupported operator')
            if n[0] in ('div','fn') and __import__('os').environ.get('AST_BPE_BROAD_STRUCTURES')!='1':raise ValueError('Pilot restricts rational and transcendental rollout domains')
            def has_variable(child):
                return child[0]=='v' or any(has_variable(v) for v in child[1:] if isinstance(v,tuple))
            if n[0] in ('d','lap','grad','divergence') and not has_variable(n[1]):
                raise ValueError('Derivative of a constant is zero and cannot identify a coefficient')
            for c in n[1:]:
                if isinstance(c,tuple):check(c)
        check(node)
        for atom in terms(node):expand_laplacian(atom,spatial)
        rhs.append(node)
    return System(tuple(fields),tuple(rhs),text,tuple(spatial))

def design(rhs,trajectory,field):
    """Build a candidate design matrix using per-trajectory AST-term caching.
    Candidate formulas reuse the same derivative atoms; cache only stores finite
    flattened columns and is cleared between trajectories, not between candidates.
    """
    y=np.asarray(trajectory['targets'][field],dtype=float)
    spatial=trajectory['spatial']; cache=trajectory.setdefault('_design_term_cache',{})
    columns=[]
    for term in terms(rhs):
        atom=expand_laplacian(term,spatial);key=repr(atom)
        col=cache.get(key)
        if col is None:
            col=np.broadcast_to(evaluate(atom,trajectory['values'],trajectory['derivative']),y.shape).ravel().copy()
            if not np.isfinite(col).all():raise ValueError('Nonfinite design')
            if col.nbytes<=128*1024*1024:
                while cache and sum(v.nbytes for v in cache.values())+col.nbytes>128*1024*1024:cache.pop(next(iter(cache)))
                cache[key]=col
        columns.append(col)
    X=np.column_stack(columns)
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite design')
    return X,y.ravel()

def solve(A,b):
    A=np.vstack(A);b=np.concatenate(b);norms=np.linalg.norm(A,axis=0)
    if np.any(norms<1e-14):raise ValueError('Zero candidate column')
    coef,_,rank,_=np.linalg.lstsq(A/norms,b,rcond=None)
    if rank!=A.shape[1]:raise ValueError('Duplicate or collinear columns')
    return coef/norms

def reward(system,trajectories,parameter_basis=('1',)):
    return fit(system,trajectories,parameter_basis,design,solve,epi=.2)
