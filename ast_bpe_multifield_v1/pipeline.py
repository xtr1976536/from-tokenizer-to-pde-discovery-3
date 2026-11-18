"""GRE-free candidate generation and whole-system reward.
The existing Transformer forward signature (logits, auxiliary) is preserved.
No new encoder, no imported GRE dictionary, no unrestricted Python evaluation.
"""
from dataclasses import dataclass
import numpy as np
from ast_runtime import LatexParser,expand_laplacian,fit_score,fit_score_multi,terms

SEPARATOR = r'\\'
@dataclass
class System:
    fields: tuple
    rhs: tuple
    text: str

def parse_system(text, fields=('u','v'), spatial=('x','y'), parameters=(), functions=()):
    if len(set(fields))!=len(fields) or not fields:raise ValueError('Unique fields required')
    segments=text.split(SEPARATOR)
    if len(segments)!=len(fields) or any(not s.strip() for s in segments):raise ValueError('Wrong equation count')
    nodes=[]
    for segment in segments:
        parser_args=dict(fields=fields,parameters=parameters,functions=functions,coordinates=spatial+('t',))
        if '=' in segment:
            left,right=segment.split('=',1)
            lhs=LatexParser(left,**parser_args).parse();rhs=LatexParser(right,**parser_args).parse()
            target=('d',('v',fields[len(nodes)]),'t')
            parts=terms(lhs)
            if parts.count(target)!=1:raise ValueError('Expected one unit target time derivative on LHS')
            rest=list(parts);rest.remove(target)
            node=('add',rhs,*(('neg',t) for t in rest)) if rest else rhs
        else:
            node=LatexParser(segment,**parser_args).parse()
        def check(n):
            if n[0]=='d' and ('t' in n[2] or any(a not in spatial for a in n[2])):raise ValueError('RHS time derivatives/undeclared axes forbidden')
            for x in n[1:]:
                if isinstance(x,tuple):check(x)
        check(node);nodes.append(expand_laplacian(node,spatial))
    return System(tuple(fields),tuple(nodes),text)

def score_system(system, trajectory, ridge=0):
    results=[]
    for field,rhs in zip(system.fields,system.rhs):
        if isinstance(trajectory,(list,tuple)):
            r=fit_score_multi(rhs,trajectory,field,ridge)
        else:
            r=fit_score(rhs,trajectory['targets'][field],trajectory['values'],trajectory['derivative'],ridge,
                        trajectory['fit_mask'],trajectory['score_mask'])
        results.append(dict(field=field,**r))
    error=float(np.mean([r['relative_mse'] for r in results]))
    return dict(status='evaluated',reward=1/(1+error),relative_mse=error,equations=results)

class ASTAdapter:
    """Control IDs never enter the HardASTBPE byte vocabulary."""
    pad,eos,bos=0,1,2
    def __init__(self,tokenizer):self.tokenizer=tokenizer;self.vocab_size=len(tokenizer.vocab)+3
    def sequence(self,text):return [self.bos]+[i+3 for i in self.tokenizer.encode(text)]+[self.eos]
    def decode(self,ids):
        if any(i<3 for i in ids):raise ValueError('Control token in expression')
        return self.tokenizer.decode([i-3 for i in ids])

def generate(model,adapter,fields=('u','v'),spatial=('x','y'),max_tokens=256,seed=525,parameters=()):
    import torch
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    device=next(model.parameters()).device;prefix=[adapter.bos];model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            x=torch.tensor([prefix],device=device)
            logits,_=model(x);p=logits.reshape(1,len(prefix),adapter.vocab_size)[0,-1].softmax(-1)
            # Preserve the current unconstrained 80/20 probability/uniform policy.
            if torch.rand((),device=device,generator=generator)<=.8:
                token=int(torch.multinomial(p,1,generator=generator))
            else:
                valid=torch.nonzero(p!=0).flatten();token=int(valid[torch.randint(len(valid),(1,),device=device,generator=generator)])
            if token==adapter.eos:
                return parse_system(adapter.decode(prefix[1:]),fields,spatial,parameters)
            if token in (adapter.bos,adapter.pad):raise ValueError('Unexpected control')
            prefix.append(token)
    raise ValueError('Unterminated system')

def search_round(propose,trajectory,target=400,max_attempts=None,previous=()):
    """An explicit failure is returned if success budget cannot be filled."""
    scored=[];attempts=0;rejections={}
    while len(scored)<target and (max_attempts is None or attempts<max_attempts):
        attempts+=1
        try:
            system=propose(attempts);r=score_system(system,trajectory)
            scored.append(dict(text=system.text,**r))
        except (ValueError,KeyError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:
            k=type(e).__name__;rejections[k]=rejections.get(k,0)+1
    if len(scored)!=target:raise RuntimeError(f'Budget incomplete: {len(scored)}/{target}, {attempts} attempts')
    return dict(top10=select_top(list(previous)+scored),attempts=attempts,evaluated=len(scored),unique_systems=len({r['text'] for r in scored}),rejections=rejections)

def select_top(rows):
    # Preserve the mainline float32 reward ranking and reward-value deduplication.
    selected=[];rewards=set()
    for original in sorted(rows,key=lambda r:float(np.float32(r['reward'])),reverse=True):
        r=dict(original);r['reward']=float(np.float32(r['reward']))
        if r['reward'] not in rewards:
            selected.append(r);rewards.add(r['reward'])
        if len(selected)==10:break
    return selected

def continuation_sequences(report,adapter,max_positions):
    seq=[adapter.sequence(r['text']) for r in report['top10']]
    if any(len(s)-1>max_positions for s in seq):raise ValueError('Whole-system sequence exceeds model capacity')
    return seq

def continue_training(model,adapter,report,optimizer,max_positions,epochs=5):
    """Same teacher-forced next-token cross-entropy as the current mainline.
    Each row is a whole ordered system including the separator, BOS and EOS.
    """
    import torch
    seqs=continuation_sequences(report,adapter,max_positions)
    if not seqs:return []
    device=next(model.parameters()).device
    batch=torch.full((len(seqs),max(map(len,seqs))),adapter.pad,device=device,dtype=torch.long)
    for i,s in enumerate(seqs):batch[i,:len(s)]=torch.tensor(s,device=device)
    model.train();losses=[]
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(batch[:,:-1])
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,adapter.vocab_size),batch[:,1:].reshape(-1),ignore_index=adapter.pad)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite continuation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        losses.append(float(loss.detach()))
    return losses
