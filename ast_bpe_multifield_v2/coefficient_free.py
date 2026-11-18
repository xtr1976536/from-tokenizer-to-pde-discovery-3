"""Parse first, then normalize scalar coefficient factors without deleting fields.

Unspecified functions and exponent parameters are rejected, never filled in.
Numeric differentiation orders and powers are retained as mathematical structure.
"""
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'ast_bpe_multifield_v1'))
from ast_runtime import LatexParser, Unsupported, FUNCS, expand_laplacian

PARAM=re.compile(r'(?:\\[A-Za-z]+|[A-Za-z])(?:_\{[^{}]*\}|_[A-Za-z0-9]+)?')
OPERATORS=set(r'\partial \frac \dfrac \tfrac \Delta \nabla \cdot \times'.split())|{'\\'+s for s in FUNCS}
GREEK=set(('\\'+s) for s in 'alpha beta gamma delta epsilon varepsilon chi xi eta theta mu nu rho sigma tau omega lambda kappa phi varphi psi pi Phi Psi Gamma Lambda zeta'.split())


class Parser(LatexParser):
    def atom(self,scripts=True):
        value=super().atom(scripts=False)
        if scripts and self.peek()=='_':
            self.take();value=('d',value,self.derivative_variables())
        if scripts and self.peek()=='^':
            self.take();value=('pow',value,self.exponent())
        return value


def parse(text,fields,scalar_parameters=()):
    text=re.sub(r'\\(?:biggl|biggr|bigl|bigr|Big|big)\b','',text)
    # Some papers redundantly write partial^2_{xx}; xx already specifies the
    # two differentiations. Never multiply the order twice.
    text=re.sub(r'\\partial_\{([xyzrt]{2,8})\}\^\{?([2-8])\}?',
                lambda m:r'\partial_{'+m[1]+'}' if len(m[1])==int(m[2]) else m[0],text)
    text=re.sub(r'\\partial\^\{?([2-8])\}?_\{([xyzrt]{2,8})\}',
                lambda m:r'\partial_{'+m[2]+'}' if len(m[2])==int(m[1]) else m[0],text)
    text=text.replace(r'\nabla^2',r'\Delta').replace(r'\triangle',r'\Delta')
    # Known numerical calculus notation only. No implicit macro expansion.
    names=[]
    for match in PARAM.finditer(text):
        name=match.group()
        if name in OPERATORS or name in fields or name in 'xyzrt':continue
        base=name.split('_')[0]
        if base in OPERATORS:continue
        if base in fields:
            continue
        if name.startswith('\\') and base not in GREEK:raise Unsupported('Unknown macro: '+name)
        tail=text[match.end():]
        if re.match(r'\s*\(',tail) and name not in ('e',) and name not in scalar_parameters:
            # Greek prefactors multiplying a nontrivial group are scalar
            # coefficients; a single-field argument may denote a sensitivity.
            group=re.match(r'\s*\(([^()]*)\)',tail)
            if group and name in GREEK and all(c in 'xyzrt, ' for c in group[1]):
                # Spatial/temporal coefficient functions are a single nuisance
                # coefficient atom, never "beta times x".
                names.append(name+group[0]);continue
            if base not in GREEK or group and group[1].strip() in fields:
                raise Unsupported('Unspecified function: '+name)
        if name=='i':raise Unsupported('Complex field system requires explicit real/imaginary decomposition')
        names.append(name)
    parameters=tuple(sorted(set(names),key=len,reverse=True))
    ast=Parser(text,fields=fields,coordinates=tuple('xyzrt'),parameters=parameters).parse()
    return ast,parameters


def contains_parameter(node,params):
    return node[0]=='v' and node[1] in params or any(contains_parameter(c,params) for c in node[1:] if isinstance(c,tuple))

def constant_expression(node,params):
    if node[0]=='n':return True
    if node[0]=='v':return node[1] in params
    return node[0] in ('neg','add','mul','div','pow') and all(constant_expression(c,params) for c in node[1:] if isinstance(c,tuple))


def normalize(node,params,coefficient=False):
    kind=node[0]
    if coefficient and constant_expression(node,params):
        return ('n',0.) if node==('n',0.) else ('n',1.)
    if kind=='v' and node[1] in params:return ('n',1.)
    if kind=='n':return ('n',1.) if coefficient and node[1]!=0 else node
    if kind=='pow':
        if contains_parameter(node[2],params):raise Unsupported('Exponent parameter needs explicit source specification')
        if node[1]==('v','e'):return ('fn','exp',normalize(node[2],params))
        return ('pow',normalize(node[1],params),node[2])
    if kind=='mul':
        parts=[normalize(c,params,True) for c in node[1:]]
        parts=[c for c in parts if c!=('n',1.)]
        return ('n',1.) if not parts else parts[0] if len(parts)==1 else ('mul',*parts)
    if kind=='div':
        a,b=normalize(node[1],params,True),normalize(node[2],params,True)
        return a if b==('n',1.) else ('div',a,b)
    if kind in ('grad','divergence','lap','d','neg','add','fn'):
        return tuple(normalize(c,params) if isinstance(c,tuple) else c for c in node)
    if kind in ('n','v'):return node
    raise Unsupported('Unsupported AST operator: '+kind)


def latex(n):
    k=n[0]
    if k=='n':return str(int(n[1])) if n[1]==int(n[1]) else str(n[1])
    if k=='v':return n[1]
    if k=='neg':return '-'+('('+latex(n[1])+')' if n[1][0]=='add' else latex(n[1]))
    if k=='add':
        output=latex(n[1])
        for child in n[2:]:output+=('' if child[0]=='neg' else '+')+latex(child)
        return output
    if k=='mul':return r'\cdot '.join('('+latex(c)+')' if c[0] in ('add','neg') else latex(c) for c in n[1:])
    if k=='div':return r'\frac{('+latex(n[1])+')}{'+latex(n[2])+'}'
    if k=='pow':return (latex(n[1]) if n[1][0] in ('v','n') else '('+latex(n[1])+')')+'^{'+latex(n[2])+'}'
    if k=='d':return r'\partial_{'+n[2]+'}('+latex(n[1])+')'
    if k=='lap':return r'\Delta'+('^{'+str(n[2])+'}' if n[2]>1 else '')+'('+latex(n[1])+')'
    if k=='grad':return r'\nabla('+latex(n[1])+')'
    if k=='divergence':return r'\nabla\cdot('+latex(n[1])+')'
    if k=='fn':return '\\'+n[1]+'('+latex(n[2])+')'
    raise Unsupported(k)
