"""Training-only LLM structure search with numerical feedback and student adaptation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import runpy

import numpy as np
import torch
from distill_models import DistillStudent, make_student
from train_distill import load_cache, integrate_batch
from symbolic_dynamics import SymbolicDynamics, additive_terms, evaluate
from candidate_rules import fit_coefficients, canonical_system_key, validate_system
from llm_equation_proposer import propose, feedback_prompt


def write(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_student(path, device):
    c = torch.load(path, map_location='cpu', weights_only=False)
    m = make_student(*c['shape'], c['dim'], c['width'],c.get('architecture','compact')).to(device)
    m.load_state_dict(c['state']); m.eval()
    return m, c


def norm_input(x, norm, device):
    x = torch.from_numpy(np.array(x, copy=True)).float().permute(0, 1, 4, 2, 3).to(device)
    if norm:
        mean = x.new_tensor(norm['mean_input']).view(1, 1, 2, 1, 1)
        std = x.new_tensor(norm['std_input']).clamp_min(1e-8).view(1, 1, 2, 1, 1)
        x = (x - mean) / std
    return x.flatten(1, 2)


def encode(data, model, ids, norm, device, trajectory_batch=32, window_batch=256):
    trajectories = None
    if getattr(model,'architecture','compact')=='unet_spatial_ode':trajectory_batch,window_batch=4,4
    with torch.inference_mode():
        for start in range(0, len(ids), trajectory_batch):
            chunk = ids[start:start+trajectory_batch]
            sequences = np.concatenate((data['x'][chunk], data['truth'][chunk]), axis=1)
            window_count = len(chunk)*21
            encoded = []
            offsets = np.arange(20)
            for i in range(0, window_count, window_batch):
                indices = np.arange(i, min(i+window_batch, window_count))
                windows = sequences[(indices//21)[:,None], (indices%21)[:,None]+offsets[None,:]]
                encoded.append(model.encoder(norm_input(windows, norm, device)).cpu())
            encoded=torch.cat(encoded)
            shaped=encoded.reshape(len(chunk),21,*encoded.shape[1:])
            if trajectories is None:trajectories=torch.empty((len(ids),*shaped.shape[1:]),dtype=encoded.dtype)
            trajectories[start:start+len(chunk)] = shaped
            if (start+len(chunk)) % (32 if getattr(model,'architecture','compact')=='unet_spatial_ode' else 512) < trajectory_batch:
                print('ENCODED', start+len(chunk), '/', len(ids), flush=True)
    return trajectories


def field_mse(data, model, dyn, ids, norm, device, batch=2):
    total = None; count = 0
    if getattr(model,'architecture','compact')=='unet_spatial_ode':batch=1
    with torch.inference_mode():
        for start in range(0, len(ids), batch):
            chunk = ids[start:start+batch]
            x = norm_input(data['x'][chunk], norm, device)
            z = model.encoder(x)
            grids=np.asarray(data['times'][chunk])
            zs = integrate_batch(dyn,z,grids,check_finite=False)
            h, w = data['x'].shape[2:4]
            pred = model(zs.flatten(0, 1)).reshape(len(chunk), 20, 2, h, w)
            if norm:
                pred = pred * pred.new_tensor(norm['std_target']).view(1,1,2,1,1) + pred.new_tensor(norm['mean_target']).view(1,1,2,1,1)
            truth = torch.from_numpy(np.array(data['truth'][chunk], copy=True)).permute(0,1,4,2,3).to(device)
            error = (pred-truth).square()
            if not torch.isfinite(error).all(): raise FloatingPointError('Nonfinite decoded prediction')
            batch_total=error.sum()
            total=batch_total if total is None else total+batch_total
            count += error.numel()
    return float(total)/count


def field_score_batch_size(batch_size):
    return max(64, batch_size)


def fit_candidate(text, z, times, fit, hold, fields, parse_system, device,
                  batch_size=4096, substeps=4):
    if z.ndim>3:batch_size=min(batch_size,64)
    started=time.monotonic()
    system = parse_system(text, fields=fields, spatial=())
    validate_system(system)
    states = z[fit].reshape(-1,z.shape[-1])
    derivatives = torch.from_numpy(np.stack([np.gradient(z[i].numpy(), times[i], axis=0, edge_order=2) for i in fit])).reshape(-1,z.shape[-1])
    coefficients = []
    for j, rhs in enumerate(system.rhs):
        A = torch.stack([evaluate(n, states, fields) for n in additive_terms(rhs)], -1).numpy()
        coefficients.append(fit_coefficients(A, derivatives[:,j].numpy()).tolist())
    fit_seconds=time.monotonic()-started
    dyn = SymbolicDynamics(system, coefficients).to(device)
    total = None; count = 0
    if batch_size < 1:
        raise ValueError('Invalid rollout batching configuration')
    with torch.inference_mode():
        for start in range(0,len(hold),batch_size):
            chunk=hold[start:start+batch_size]
            pred = integrate_batch(dyn,z[chunk,0].to(device),times[chunk],
                                   substeps=substeps,check_finite=False)
            error = (pred-z[chunk,1:].to(device)).square()
            if not torch.isfinite(error).all(): raise FloatingPointError('Nonfinite latent score')
            batch_total=error.sum()
            total=batch_total if total is None else total+batch_total
            count += error.numel()
    rollout_seconds=time.monotonic()-started-fit_seconds
    return dict(text=text, fields=fields, coefficients=coefficients,
                structural_family_key=canonical_system_key(system), status='PASS',
                internal_holdout_latent_rollout_mse=float(total)/count,
                coefficient_fit_seconds=fit_seconds,latent_rollout_seconds=rollout_seconds,
                source='LLM proposal', training_only=True, external_validation_used=False), dyn


def main():
    p = argparse.ArgumentParser()
    for name in ('root','cache','output'): p.add_argument('--'+name, required=True)
    initialization=p.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--student')
    initialization.add_argument('--from-scratch',action='store_true')
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--architecture',choices=('compact','unet_spatial_ode'),default='compact')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--lr-min',type=float,default=1e-6)
    p.add_argument('--lambda-teacher',type=float,default=1.)
    p.add_argument('--lambda-real',type=float,default=1.)
    p.add_argument('--dim',type=int,choices=(2,4,8),default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cycles', type=int, default=5)
    p.add_argument('--alternating',action='store_true',help='Search again before every complete teacher-supervised training epoch')
    p.add_argument('--calls-per-cycle', type=int, default=3)
    p.add_argument('--candidates-per-call', type=int, default=8)
    p.add_argument('--shortlist', type=int, default=8)
    p.add_argument('--adaptation-updates', type=int, default=100)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--gpu-memory-gib', type=float, default=20)
    p.add_argument('--model', default='gpt-5.6')
    p.add_argument('--base-url', default='https://www.aiwanwu.cc/v1')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume-run', action='store_true')
    a = p.parse_args()
    if min(a.cycles,a.calls_per_cycle,a.candidates_per_call,a.shortlist,a.adaptation_updates,a.epochs,a.batch_size)<1: raise ValueError('Budgets must be positive')
    torch.set_num_threads(a.threads); torch.manual_seed(525); np.random.seed(525)
    if a.device.startswith('cuda'):
        total = torch.cuda.get_device_properties(a.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,a.gpu_memory_gib*1024**3/total), a.device)
    root = Path(a.root).resolve()
    sys.path.insert(0,str(root/'research/realpdebench/upstream'))
    sys.path.insert(0, str(root/'ast_bpe_multifield_v1'))
    from pipeline import parse_system
    out = Path(a.output); out.mkdir(parents=True, exist_ok=a.resume_run)
    if a.resume_run:
        previous=json.loads((out/'config.json').read_text())
        if any(previous.get(k,p.get_default(k))!=v for k,v in vars(a).items() if k!='resume_run'):
            raise ValueError('Resume configuration differs from the original run')
    else: write(out/'config.json', vars(a))
    manifest = json.loads((Path(a.cache)/'manifest.json').read_text())
    data = load_cache(a.cache,include_teacher=a.lambda_teacher!=0)
    counts = {s:int(np.count_nonzero(data['split']==s)) for s in ('train','val','test')}
    if counts != {'train':8147,'val':832,'test':0}: raise ValueError(f'Unexpected formal cache split: {counts}')
    norm = manifest['normalization']
    train = np.flatnonzero(data['split']=='train')
    groups = np.asarray(data['trajectory_group']).astype(str)
    unique = sorted(set(groups[train])); fit_groups = set(unique[::2])
    if len(unique)<2: raise ValueError('Need distinct training trajectory groups')
    fit_ids = train[np.isin(groups[train], list(fit_groups))]
    hold_ids = train[~np.isin(groups[train], list(fit_groups))]
    cache_path = a.cache
    if a.smoke:
        # Real full-resolution data; small subset includes both grouped partitions.
        train = np.concatenate((fit_ids[:4],hold_ids[:4]))
        val = np.flatnonzero(data['split']=='val')[:4]
        selected = np.concatenate((train,val))
        small = out/'smoke_cache'
        if not small.exists():
            small.mkdir()
            for key in (('x','teacher','truth','times','split') if a.lambda_teacher!=0 else ('x','truth','times','split')): np.save(small/(key+'.npy'),data[key][selected])
            sm = dict(manifest); sm['entries'] = [manifest['entries'][i] for i in selected]
            write(small/'manifest.json',sm)
        data = load_cache(small,include_teacher=a.lambda_teacher!=0); cache_path = str(small)
        train = np.flatnonzero(data['split']=='train'); groups = np.asarray(data['trajectory_group']).astype(str)
    fit = [k for k,i in enumerate(train) if groups[i] in fit_groups]
    hold = [k for k,i in enumerate(train) if groups[i] not in fit_groups]
    if not fit or not hold: raise ValueError('Empty internal partition')
    if a.from_scratch:
        initial=out/'random_initial_student.pt'
        if not a.resume_run:
            h,w=data['x'].shape[2:4]
            model=make_student(h,w,a.dim,a.width,a.architecture)
            torch.save(dict(state=model.state_dict(),shape=[h,w],dim=a.dim,width=a.width,
                       rhs_width=128,architecture=a.architecture,normalization=norm,dynamics_kind='random_untrained',
                       initialization='random_seed_525',pretrained_weights_used=False,
                       training_updates=0),initial)
            del model
        initial_checkpoint=torch.load(initial,map_location='cpu',weights_only=False)
        if initial_checkpoint.get('training_updates')!=0 or initial_checkpoint.get('pretrained_weights_used') is not False:
            raise ValueError('Scratch initialization must be an untrained random student')
        current=str(initial.resolve())
    else:
        current=str(Path(a.student).resolve())
    write(out/'protocol.json',dict(counts=counts,smoke=a.smoke,fit_groups=sorted(fit_groups),holdout_groups=sorted(set(unique)-fit_groups),
          initialization='random_untrained' if a.from_scratch else 'checkpoint',
          architecture=a.architecture,latent_dimension_meaning='spatial feature channels' if a.architecture=='unet_spatial_ode' else 'global vector dimension',
          learning_rate_schedule='global epoch cosine' if a.architecture=='unet_spatial_ode' else 'constant',
          lr_start=a.lr,lr_min=a.lr_min,effective_batch_size=a.batch_size,
          training_schedule='numerical selection without student updates then full teacher epoch',
          candidate_training_updates=0,
          maximum_alternating_epochs=a.epochs if a.alternating else None,
          pretrained_student_used=not a.from_scratch,teacher_supervision='frozen cached teacher predictions plus real training labels' if a.lambda_teacher else 'none: real labels only',
          lambda_teacher=a.lambda_teacher,lambda_real=a.lambda_real,
          feedback_source='training trajectories only',test_targets_used=False,student_sha256=digest(current),cache_manifest_sha256=digest(Path(a.cache)/'manifest.json')))
    history=[]
    if a.resume_run and (out/'progress.json').exists():
        history=json.loads((out/'progress.json').read_text()); current=history[-1]['student']
    for cycle in range(a.epochs):
        if cycle<len(history): continue
        if history and a.patience and history[-1]['stale_cycles']>=a.patience:
            break
        cd = out/f'cycle_{cycle+1}'; cd.mkdir(exist_ok=a.resume_run)
        model, checkpoint = load_student(current,a.device)
        if checkpoint['dim'] != a.dim: raise ValueError('Checkpoint latent dimension mismatch')
        if checkpoint.get('normalization')!=norm:
            raise ValueError('Checkpoint/cache normalization mismatch')
        fields = tuple('uvwpqrsz'[:a.dim]); z = encode(data,model,train,norm,a.device)
        np.save(cd/'latent_trajectories.npy',z.numpy())
        times = np.asarray(data['times'][train])
        # Fixed samples from both training partitions, never external validation.
        sample = sorted(set(np.linspace(0,len(train)-1,min(8,len(train)),dtype=int).tolist()))
        summary = dict(coordinate_version=digest(current),normalization=norm,
            latent_layout='spatial_channels' if z.ndim>3 else 'global_vector',
            spatial_shape=list(z.shape[2:-1]),training_windows=len(train),latent_min=z.reshape(-1,a.dim).amin(0).tolist(),latent_max=z.reshape(-1,a.dim).amax(0).tolist(),
            trajectories=[dict(times=times[i].tolist(),states=(z[i].mean(tuple(range(1,z[i].ndim-1))).tolist() if z[i].ndim>2 else z[i].tolist())) for i in sample])
        write(cd/'data_summary.json',summary)
        feedback=([history[-1]['post_adaptation']] if history else []); seen=set(); passing=[]
        for call in range(a.calls_per_cycle):
            prompt = feedback_prompt(dimension=a.dim,summaries=summary,prior=[r['text'] for r in passing],feedback=feedback,count=a.candidates_per_call)
            (cd/f'prompt_{call+1}.json').write_text(prompt)
            started=time.monotonic()
            response_path=cd/f'response_{call+1}.json'
            if response_path.exists(): rows=json.loads(response_path.read_text())['systems']
            else:
                rows=propose(prompt,model=a.model,base_url=a.base_url,audit_path=cd/f'api_{call+1}.json')
                write(response_path,dict(systems=rows,seconds=time.monotonic()-started))
            for row in rows[:a.candidates_per_call]:
                text = row.get('text') if isinstance(row,dict) else None
                try:
                    if not isinstance(text,str): raise ValueError('Missing equation text')
                    system=parse_system(text,fields=fields,spatial=()); validate_system(system)
                    key=canonical_system_key(system)
                    if key in seen: raise ValueError('Duplicate structure')
                    seen.add(key)
                    started=time.monotonic()
                    report,dyn=fit_candidate(text,z,times,fit,hold,fields,parse_system,a.device)
                    report['fit_and_rollout_seconds']=time.monotonic()-started
                    print('CANDIDATE_EVALUATED',cycle+1,len(seen),
                          'status=PASS','seconds',round(report['fit_and_rollout_seconds'],3),flush=True)
                    field_started=time.monotonic()
                    report['internal_holdout_field_mse_before_adaptation']=field_mse(data,model,dyn,train[hold],norm,a.device,field_score_batch_size(a.batch_size))
                    report['field_score_seconds']=time.monotonic()-field_started
                    passing.append(report); feedback.append(report)
                except (ValueError,KeyError,TypeError,RuntimeError,FloatingPointError) as exc:
                    feedback.append(dict(text=text,status='REJECTED',error=str(exc)[:300]))
            write(cd/'feedback.json',feedback)
            print('LLM_CALL',cycle+1,call+1,'passing',len(passing),flush=True)
        if not passing: raise RuntimeError('No LLM candidate survived numerical checks')
        # Candidate selection never updates encoder or decoder parameters.
        passing.sort(key=lambda r:(r['internal_holdout_field_mse_before_adaptation'],
                                   r['internal_holdout_latent_rollout_mse']))
        del model
        for chosen in passing:
            try:
                artifact=cd/'selected_equation.json';write(artifact,chosen)
                print('NUMERICAL_SELECTION',cycle+1,'field_mse',chosen['internal_holdout_field_mse_before_adaptation'],
                      'candidate_training_updates',0,flush=True)
                epoch_dest=cd/'teacher_epoch'
                epoch_cmd=[str(Path(__file__).with_name('train_distill.py')),
                    '--cache',cache_path,'--resume',current,'--continue-optimizer','--refit-equation',
                    '--equation-artifact',str(artifact),'--runtime-root',str(root),
                    '--output',str(epoch_dest),'--epochs','1','--seed',str(525+cycle),
                    '--dim',str(a.dim),'--width',str(checkpoint['width']),'--rhs-width',str(checkpoint['rhs_width']),
                    '--batch-size',str(a.batch_size),'--device',a.device,
                    '--lambda-teacher',str(a.lambda_teacher),'--lambda-real',str(a.lambda_real)]
                if a.architecture=='unet_spatial_ode':
                    import math
                    lr=a.lr_min+(a.lr-a.lr_min)*(1+math.cos(math.pi*cycle/max(1,a.epochs-1)))/2
                    epoch_cmd+=['--architecture',a.architecture,'--batch-size','1',
                                '--gradient-accumulation',str(a.batch_size),'--lr',str(lr),'--override-lr']
                print('TEACHER_FULL_EPOCH_START',cycle+1,flush=True)
                original=sys.argv;sys.argv=epoch_cmd
                try:
                    record=epoch_dest/'training.json'
                    complete=record.exists() and json.loads(record.read_text()).get('status')=='COMPLETE' and (epoch_dest/'student.pt').exists()
                    if not complete:
                        if epoch_dest.exists():
                            epoch_dest.rename(epoch_dest.with_name(epoch_dest.name+f'_interrupted_{time.time_ns()}'))
                        runpy.run_path(epoch_cmd[0],run_name='__main__')
                finally:sys.argv=original
                epoch_record=json.loads((epoch_dest/'training.json').read_text())
                if not epoch_record['history'][-1]['full_training_epoch']:
                    raise RuntimeError('Teacher epoch must visit every training window')
                current=str(epoch_dest/'student.pt')
                break
            except FloatingPointError as exc:
                print('CANDIDATE_TRAINING_DIVERGED',cycle+1,str(exc),flush=True)
                write(cd/f"diverged_{time.time_ns()}.json",dict(text=chosen['text'],error=str(exc)))
        else:
            raise RuntimeError('All numerically screened candidates diverged during teacher training')
        selected_model,selected=load_student(current,a.device)
        equation=selected['equation_artifact']
        selected_dyn=SymbolicDynamics(parse_system(equation['text'],fields=fields,spatial=()),equation['coefficients']).to(a.device)
        selected_dyn.load_state_dict(selected['rhs'])
        post=field_mse(data,selected_model,selected_dyn,train[hold],norm,a.device,field_score_batch_size(a.batch_size))
        summary_post=dict(coordinate_version=digest(current),internal_holdout_field_mse_after_epoch=post)
        write(cd/'post_epoch_feedback.json',summary_post)
        previous=history[-1] if history else {}
        choices=[(epoch_record['best_val'],current)]
        if previous:choices.append((previous['best_validation_mse'],previous['best_student']))
        best_score,best_student=min(choices,key=lambda pair:pair[0])
        meaningful=previous.get('meaningful_best',float('inf'))
        improved=best_score<meaningful-1e-6
        row=dict(cycle=cycle+1,student=current,passing=len(passing),post_adaptation=summary_post,
                 selected_equation=chosen,selection_source='training internal holdout field MSE',
                 candidate_training_updates=0,teacher_epoch=epoch_record,completed_full_epochs=cycle+1,
                 best_validation_mse=best_score,best_student=best_student,
                 meaningful_best=best_score if improved else meaningful,
                 stale_cycles=0 if improved else previous.get('stale_cycles',0)+1)
        print('ALTERNATING_EPOCH_COMPLETE',cycle+1,'validation',epoch_record['best_val'],
              'best',best_score,'stale_cycles',row['stale_cycles'],flush=True)
        history.append(row);write(out/'progress.json',history)
        del selected_model,selected_dyn,z,dyn
    last=history[-1];selected=last['best_student']
    checkpoint=torch.load(selected,map_location='cpu',weights_only=False)
    write(out/'selected_equation.json',checkpoint['equation_artifact'])
    write(out/'result.json',dict(status='SMOKE_COMPLETE' if a.smoke else 'COMPLETE',
          mode='numerical_selection_then_full_teacher_epoch',student=selected,student_sha256=digest(selected),
          best_validation_mse=last['best_validation_mse'],completed_full_epochs=len(history),
          early_stopped=len(history)<a.epochs,history=history,test_targets_used=False))
    print('ALTERNATING_TRAINING_COMPLETE',selected,flush=True)


if __name__=='__main__': main()
