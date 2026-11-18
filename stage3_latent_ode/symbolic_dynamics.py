"""Differentiable explicit ODE RHS; never clips or silently repairs domains."""
import torch
from torch import nn


def evaluate(node, z, fields):
    op = node[0]
    if op == 'n':
        return z.new_full(z.shape[:-1], float(node[1]))
    if op == 'v':
        return z[..., fields.index(node[1])]
    if op == 'neg':
        return -evaluate(node[1], z, fields)
    args = [evaluate(n, z, fields) for n in node[1:] if isinstance(n, tuple)]
    if op == 'add':
        return torch.stack(args).sum(0)
    if op == 'mul':
        return torch.stack(args).prod(0)
    if op == 'div':
        return args[0] / args[1]
    if op == 'pow':
        return args[0] ** args[1]
    if op == 'fn':
        functions = {'sin': torch.sin, 'cos': torch.cos, 'tanh': torch.tanh,
                     'exp': torch.exp, 'log': torch.log, 'sqrt': torch.sqrt,
                     'abs': torch.abs, 'sinh': torch.sinh, 'cosh': torch.cosh,
                     'tan': torch.tan}
        if node[1] not in functions:
            raise ValueError('Unsupported ODE function: ' + node[1])
        return functions[node[1]](args[0])
    raise ValueError('Unsupported ODE AST operator: ' + op)


def additive_terms(node):
    if node[0] == 'add':
        return [t for child in node[1:] for t in additive_terms(child)]
    return [node]


class SymbolicDynamics(nn.Module):
    def __init__(self, system, coefficients):
        super().__init__()
        self.fields = tuple(system.fields)
        self.nodes = [additive_terms(rhs) for rhs in system.rhs]
        if len(coefficients) != len(self.nodes):
            raise ValueError('Coefficient/equation count mismatch')
        if any(len(c) != len(n) for c, n in zip(coefficients, self.nodes)):
            raise ValueError('Coefficient/term count mismatch')
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.tensor(c, dtype=torch.float32)) for c in coefficients])

    def forward(self, z):
        if z.shape[-1] != len(self.fields):
            raise ValueError('Latent dimension does not match equation fields')
        result = torch.stack([
            sum(c * evaluate(n, z, self.fields) for c, n in zip(cs, nodes))
            for cs, nodes in zip(self.coefficients, self.nodes)], -1)
        return result
