"""Stored-frame isolation and immutable experiment provenance checks."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def frame_sets(entries,offset=0,length=40):
    result=defaultdict(set)
    for row in entries:result[row['sim_id']].update(range(row['time_id']+offset,row['time_id']+offset+length))
    return result


def intersection(a,b):
    per={sid:len(values&b.get(sid,set())) for sid,values in a.items()}
    return dict(trajectory_frame_pairs=sum(per.values()),trajectories=sum(v>0 for v in per.values()),
        per_trajectory={k:v for k,v in per.items() if v})


def full_window_audit(indices,excluded_test_trajectories=()):
    parts={s:frame_sets(rows) for s,rows in indices.items()}
    checks={f'{a}40_{b}40':intersection(parts[a],parts[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
    used=sorted({r['sim_id'] for r in indices['test']}&set(excluded_test_trajectories))
    return dict(full_window_isolated=all(v['trajectory_frame_pairs']==0 for v in checks.values()) and not used,
        checks=checks,development_trajectories_in_test=used,
        counts={s:dict(windows=len(rows),trajectories=len({r['sim_id'] for r in rows})) for s,rows in indices.items()},
        preprocessing_causality_verified=False,
        limit='Disjoint processed frame identities, not proof of causal raw measurement filtering')


def clean_protocol_indices(indices,development_trajectories):
    test=[r for r in indices['test'] if r['sim_id'] not in set(development_trajectories)]
    protected=frame_sets(test)
    val=[r for r in indices['val'] if not set(range(r['time_id'],r['time_id']+40))&protected.get(r['sim_id'],set())]
    vf=frame_sets(val)
    train=[r for r in indices['train'] if not set(range(r['time_id'],r['time_id']+40))&(protected.get(r['sim_id'],set())|vf.get(r['sim_id'],set()))]
    result=dict(train=train,val=val,test=test)
    if not all(result.values()) or not full_window_audit(result,development_trajectories)['full_window_isolated']:
        raise ValueError('Clean protocol split is not fully isolated')
    return result


def train_reward_partition(entries,max_windows_per_trajectory=None):
    by=defaultdict(list)
    for row in entries:by[row['sim_id']].append(row)
    fit=[];reward=[]
    for sid,rows in sorted(by.items()):
        rows=sorted(rows,key=lambda r:r['time_id'])
        if max_windows_per_trajectory is not None and len(rows)>max_windows_per_trajectory:
            idx=np.linspace(0,len(rows)-1,max_windows_per_trajectory,dtype=int)
            rows=[rows[i] for i in idx]
        if len(rows)<2:raise ValueError('Need at least two training windows per trajectory for internal reward holdout')
        n=max(1,len(rows)//5);heldout=rows[-n:];reserved=frame_sets(heldout)[sid]
        allowed=[r for r in rows[:-n] if not set(range(r['time_id'],r['time_id']+40))&reserved]
        if not allowed:raise ValueError('No disjoint internal coefficient-fit windows for '+sid)
        fit.extend(allowed);reward.extend(heldout)
    if intersection(frame_sets(fit),frame_sets(reward))['trajectory_frame_pairs']:raise ValueError('Internal reward leakage')
    return dict(fit=fit,reward=reward)


def require_causal_preprocessing(config,workspace):
    if not config['data'].get('require_full'):return
    if config.get('allow_unverified_preprocessing'):
        return
    descriptor=config['data'].get('preprocessing_provenance')
    if not descriptor:raise RuntimeError('Formal run blocked: causal raw-to-field preprocessing has not been verified')
    path=Path(descriptor)
    if not path.is_absolute():path=Path(workspace)/path
    data=json.loads(path.read_text())
    if data.get('status')!='VERIFIED_CAUSAL' or not data.get('evidence_files'):
        raise RuntimeError('Formal run blocked: causal raw-to-field preprocessing has not been verified')
    if data.get('maximum_future_support_frames')!=0:
        raise RuntimeError('Formal run blocked: processed input depends on future measurements')
    for evidence in data['evidence_files']:
        source=Path(evidence['path'])
        if not source.is_absolute():source=Path(workspace)/source
        if hashlib.sha256(source.read_bytes()).hexdigest()!=evidence['sha256']:raise ValueError('Preprocessing evidence hash mismatch')


def frame_audit(indices):
    train=frame_sets(indices['train']);val=frame_sets(indices['val']);future=frame_sets(indices['test'],20,20)
    checks=dict(train40_test_future=intersection(train,future),val40_test_future=intersection(val,future),train40_val40=intersection(train,val))
    return dict(stored_frame_isolated=all(r['trajectory_frame_pairs']==0 for r in checks.values()),checks=checks,
        counts={s:dict(windows=len(v),trajectories=len({r['sim_id'] for r in v})) for s,v in indices.items()},
        semantics='train40 and val40 are learning/selection exposure; protect test future20; test inputs may overlap earlier allowed observations',
        limit='Audits processed-frame identities, not undocumented raw-camera smoothing support or past human development exposure')


def purge40(indices):
    future=frame_sets(indices['test'],20,20)
    val=[r for r in indices['val'] if not set(range(r['time_id'],r['time_id']+40))&future.get(r['sim_id'],set())]
    val_frames=frame_sets(val)
    train=[r for r in indices['train'] if not set(range(r['time_id'],r['time_id']+40))&(future.get(r['sim_id'],set())|val_frames.get(r['sim_id'],set()))]
    result=dict(train=train,val=val,test=indices['test'])
    if not all(result.values()) or not frame_audit(result)['stored_frame_isolated']:raise ValueError('40-frame purge failed')
    return result


def validate_config(config):
    if config.get('task')!='Real-world Training':raise ValueError('Only Real-world Training is in scope')
    if config['solver'].get('steps')!=20:raise ValueError('Main experiment requires exactly 20 future frames')
    if config.get('seed',525)!=525:raise ValueError('Current implementation uses fixed seed 525; changing only metadata is forbidden')
    if config['data'].get('require_full'):
        if not config['generator'].get('production'):raise ValueError('Full experiment requires the production formula model')
        if config['solver'].get('max_seconds') is not None:raise ValueError('Full model selection cannot depend on a hardware-specific wall-time cutoff')
        if config['search']['cycles']!=5 or config['search']['successful_systems_per_cycle']!=400 or config['search']['finalists']!=10:
            raise ValueError('Full experiment requires the declared 5x400 search and Top10 finalists')
    return True


def provenance(root):
    root=Path(root)
    files=list(root.glob('*.py'))
    files.extend(root.parent/'ast_bpe_multifield_v1'/name for name in ('model.py','pipeline.py','grid.py','ast_runtime.py'))
    files.extend([root.parent/'分词器-pde发现/eqgpt_original/code/gpt_model.py',
        root.parent/'research/realpdebench/upstream/realpdebench/utils/metrics.py',
        root.parent/'ast_bpe_multifield_v2/compact_tokenizer.py',root.parent/'ast_bpe_multifield_v2/coefficient_free.py',
        root.parent/'分词器-pde发现/src/math_tokenizer/hard_ast_2026.py'])
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def verify_provenance(record):
    for name,expected in record.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Experiment source changed after freeze: '+name)


def corpus_audit(folder):
    folder=Path(folder)
    formulas=[json.loads(s) for s in (folder/'equations.jsonl').read_text().splitlines()]
    encoded=[json.loads(s) for s in (folder/'encoded.jsonl').read_text().splitlines()]
    splits=json.loads((folder/'splits.json').read_text());by_id={r['id']:r for r in formulas}
    if len(by_id)!=200 or len(encoded)!=200:raise ValueError('Expected exactly 200 source PDE systems')
    if len({r['id'] for r in encoded})!=200:raise ValueError('Duplicate encoded formula IDs')
    for split,ids in splits.items():
        if set(ids)!={r['id'] for r in encoded if r['split']==split}:raise ValueError('Encoded formula split differs from manifest')
    groups={s:{by_id[r['id']]['source_group'] for r in encoded if r['split']==s} for s in ('train','validation')}
    texts={s:{by_id[r['id']]['text'] for r in encoded if r['split']==s} for s in groups}
    if groups['train']&groups['validation'] or texts['train']&texts['validation']:raise ValueError('Formula training/validation contamination')
    hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('equations.jsonl','encoded.jsonl','splits.json','tokenizer300.json','tokenizer300.backend.json')}
    return dict(hashes=hashes,train=len(splits['train']),validation=len(splits['validation']),source_paper_overlap=0,exact_text_overlap=0,
        data_role='external equation-text prior, not simulated or experimental trajectory pretraining; must disclose as method prior')


def validate_generator_checkpoint(model,metadata,corpus,production=False):
    hashes=corpus['hashes']
    required={'tokenizer_sha256':hashes['tokenizer300.json'],'corpus_sha256':hashes['equations.jsonl']}
    if production:required.update(encoded_sha256=hashes['encoded.jsonl'],splits_sha256=hashes['splits.json'],tokenizer_backend_sha256=hashes['tokenizer300.backend.json'])
    for key,value in required.items():
        if metadata.get(key)!=value:raise ValueError('Formula checkpoint provenance mismatch: '+key)
    if production:
        expected=dict(d_model=768,d_ff=2048,d_k=64,d_v=64,n_layers=6,n_heads=8,vocab_size=300,positions=72,smoke=False)
        if any(model.stage2_config.get(k)!=v for k,v in expected.items()):raise ValueError('Production generator architecture mismatch')
    return True
