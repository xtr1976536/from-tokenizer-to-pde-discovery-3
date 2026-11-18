"""Train the latent ODE student against frozen U-Net rollouts and real targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from distill_models import DistillStudent, make_student, LatentRHS, distillation_loss


def load_cache(path,allow_partial=False,include_teacher=True):
    path=Path(path)
    if path.is_dir():
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('status')!='COMPLETE' and not (allow_partial and manifest.get('status')=='PREFLIGHT_PARTIAL'):raise ValueError('Full cache is not complete')
        cache={key:np.load(path/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in (('x','teacher','truth','times','split') if include_teacher else ('x','truth','times','split'))}
        cache['trajectory_group']=np.asarray([entry['sim_id'] for entry in manifest['entries']])
    else:
        cache = np.load(path, allow_pickle=False)
        sidecar=path.with_suffix('.manifest.json')
        if sidecar.exists():
            manifest=json.loads(sidecar.read_text());entries=manifest.get('entries',[])
            if len(entries)==cache['x'].shape[0]:cache={k:cache[k] for k in cache.files};cache['trajectory_group']=np.asarray([entry['sim_id'] for entry in entries])
    required = {'x', 'truth', 'times', 'split'}
    if include_teacher:required.add('teacher')
    if not required.issubset(cache.keys() if isinstance(cache,dict) else cache.files):
        raise ValueError(f'Cache must contain {sorted(required)}')
    x, truth = cache['x'],cache['truth']
    teacher=cache['teacher'] if include_teacher else None
    if x.shape[1] != 20 or truth.shape[1] != 20 or (include_teacher and teacher.shape[1] != 20):
        raise ValueError('Distillation protocol requires exactly 20 input and 20 output frames')
    if truth.ndim != 5 or (include_teacher and teacher.shape != truth.shape):
        raise ValueError('Expected matching [N,T,H,W,C] teacher and truth arrays')
    if x.shape[0] != truth.shape[0] or x.shape[-3:-1] != truth.shape[-3:-1]:
        raise ValueError('Input and target cache dimensions do not align')
    if np.asarray(cache['times']).shape[-1] != 21:
        raise ValueError('times must contain current time plus 20 future integration points')
    result={k: cache[k] for k in required}
    if 'trajectory_group' in cache:result['trajectory_group']=cache['trajectory_group']
    return result


def integrate_batch(rhs, z0, times, substeps=4, check_finite=True):
    times = np.asarray(times)
    if times.ndim == 2:
        if times.shape[0] != z0.shape[0]:
            raise ValueError('Time grid batch does not match state batch')
        intervals = np.diff(times, axis=1)
    elif times.ndim == 1:
        intervals = np.broadcast_to(np.diff(times)[None, :], (z0.shape[0], times.size-1))
    else:
        raise ValueError('times must be a 1D grid or a batch of grids')
    states = []
    z = z0
    for interval in range(intervals.shape[1]):
        h = torch.as_tensor(intervals[:, interval] / substeps, device=z.device, dtype=z.dtype).view(-1, *([1] * (z.ndim-1)))
        for _ in range(substeps):
            k1 = rhs(z)
            k2 = rhs(z + h*k1/2)
            k3 = rhs(z + h*k2/2)
            k4 = rhs(z + h*k3)
            z = z + h*(k1 + 2*k2 + 2*k3 + k4)/6
        states.append(z)
    result = torch.stack(states, dim=1)
    if check_finite and not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite latent ODE rollout')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dim', type=int, default=8)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--rhs-width', type=int, default=32)
    p.add_argument('--equation-artifact', type=Path)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--architecture',choices=('compact','unet_spatial_ode'),default='compact')
    p.add_argument('--override-lr',action='store_true')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda-teacher', type=float, default=1.)
    p.add_argument('--lambda-real', type=float, default=1.)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume',type=Path)
    p.add_argument('--refit-equation',action='store_true',help='Use newly fitted coefficients even if the structure matches the resume checkpoint')
    p.add_argument('--seed',type=int,default=525)
    p.add_argument('--continue-optimizer',action='store_true')
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--allow-partial-cache',action='store_true')
    p.add_argument('--max-updates',type=int)
    p.add_argument('--patience',type=int,default=0)
    p.add_argument('--min-delta',type=float,default=0.)
    a = p.parse_args()
    if a.gradient_accumulation<1:raise ValueError('Gradient accumulation must be positive')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    data = load_cache(a.cache,allow_partial=a.allow_partial_cache,include_teacher=a.lambda_teacher!=0)
    cache_manifest=json.loads((Path(a.cache)/'manifest.json').read_text()) if Path(a.cache).is_dir() else {}
    normalization=cache_manifest.get('normalization')
    if normalization:
        mean_x=torch.tensor(normalization['mean_input'],dtype=torch.float32).view(1,1,2,1,1)
        std_x=torch.tensor(normalization['std_input'],dtype=torch.float32).clamp_min(1e-8).view(1,1,2,1,1)
        mean_y=torch.tensor(normalization['mean_target'],dtype=torch.float32).view(1,1,2,1,1)
        std_y=torch.tensor(normalization['std_target'],dtype=torch.float32).clamp_min(1e-8).view(1,1,2,1,1)
    else:mean_x=std_x=mean_y=std_y=None
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    split = data['split']
    train_idx = np.flatnonzero(split == 'train')
    val_idx = np.flatnonzero(split == 'val')
    if not len(train_idx) or not len(val_idx):
        raise ValueError('Both train and validation windows are required')
    h, w = data['x'].shape[2:4]
    model = make_student(h, w, a.dim, a.width,a.architecture).to(a.device)
    equation_artifact = None
    if a.equation_artifact:
        import sys
        if not a.runtime_root:raise ValueError('--runtime-root required for symbolic equations')
        sys.path.insert(0,str(a.runtime_root/'ast_bpe_multifield_v1'))
        from pipeline import parse_system
        from symbolic_dynamics import SymbolicDynamics
        equation_artifact=json.loads(a.equation_artifact.read_text())
        system=parse_system(equation_artifact['text'],fields=tuple(equation_artifact['fields']),spatial=())
        if len(system.fields)!=a.dim:raise ValueError('Equation/student dimension mismatch')
        rhs=SymbolicDynamics(system,equation_artifact['coefficients']).to(a.device)
    else:
        rhs = LatentRHS(a.dim,a.rhs_width).to(a.device)
    if a.resume:
        previous=torch.load(a.resume,map_location='cpu',weights_only=False)
        model.load_state_dict(previous['state'])
        same_equation = (equation_artifact is not None and previous.get('equation_artifact') is not None
                         and equation_artifact['text'] == previous['equation_artifact']['text'])
        if 'rhs' in previous and ((same_equation and not a.refit_equation) or (equation_artifact is None and previous.get('dynamics_kind') == 'neural_warmup')):
            rhs.load_state_dict(previous['rhs'])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rhs_opt = torch.optim.AdamW(rhs.parameters(), lr=a.lr)
    if a.continue_optimizer and a.resume:
        if previous.get('optimizer') is not None:
            opt.load_state_dict(previous['optimizer'])
        if previous.get('rhs_optimizer') is not None and same_equation and not a.refit_equation:
            rhs_opt.load_state_dict(previous['rhs_optimizer'])
    if a.override_lr:
        for group in opt.param_groups:group['lr']=a.lr
        for group in rhs_opt.param_groups:group['lr']=a.lr
    best = float('inf')
    log = []
    updates = 0
    best_epoch = 0
    meaningful_best = float('inf')
    last_meaningful_epoch = 0
    for epoch in range(a.epochs):
        model.train()
        loss_total = None
        loss_batches = 0
        epoch_samples = 0
        order = np.random.permutation(train_idx)
        chunks=[order[start:start+a.batch_size] for start in range(0,len(order),a.batch_size)]
        opt.zero_grad(set_to_none=True);rhs_opt.zero_grad(set_to_none=True)
        for batch_index,ids in enumerate(chunks):
            if not len(ids):
                continue
            x=torch.from_numpy(np.asarray(data['x'][ids])).float().permute(0,1,4,2,3); truth=torch.from_numpy(np.asarray(data['truth'][ids])).float().permute(0,1,4,2,3)
            xb=x.to(a.device); yb=truth.to(a.device)
            tb=torch.from_numpy(np.asarray(data['teacher'][ids])).float().permute(0,1,4,2,3).to(a.device) if a.lambda_teacher!=0 else None
            if mean_x is not None:
                xb=(xb-mean_x.to(a.device))/std_x.to(a.device)
                if tb is not None:tb=(tb-mean_y.to(a.device))/std_y.to(a.device)
                yb=(yb-mean_y.to(a.device))/std_y.to(a.device)
            z=model.encoder(xb.reshape(len(ids), 40, h, w))
            latent=integrate_batch(rhs,z,np.asarray(data['times'][ids]))
            pred=model(latent.flatten(0,1)).reshape(len(ids),20,2,h,w)
            loss=distillation_loss(pred,tb,yb,a.lambda_teacher,a.lambda_real)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            group_start=(batch_index//a.gradient_accumulation)*a.gradient_accumulation
            group_count=sum(len(part) for part in chunks[group_start:group_start+a.gradient_accumulation])
            (loss*len(ids)/group_count).backward()
            if (batch_index+1)%a.gradient_accumulation==0 or batch_index+1==len(chunks):
                norm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(rhs.parameters()),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite training gradient')
                opt.step();rhs_opt.step();opt.zero_grad(set_to_none=True);rhs_opt.zero_grad(set_to_none=True)
                updates += 1
            detached_loss=loss.detach()
            loss_total=detached_loss if loss_total is None else loss_total+detached_loss
            loss_batches+=1
            epoch_samples += len(ids)
            if (batch_index+1)%100 == 0:
                print('STUDENT_BATCH',epoch+1,batch_index+1,'/',len(chunks),'samples',epoch_samples,'updates',updates,flush=True)
            if a.max_updates is not None and updates >= a.max_updates:
                break
        model.eval()
        with torch.inference_mode():
            val_total=None; val_count=0
            for ids in np.array_split(val_idx, max(1, (len(val_idx)+a.batch_size-1)//a.batch_size)):
                if not len(ids): continue
                xv=torch.from_numpy(np.asarray(data['x'][ids])).float().permute(0,1,4,2,3)
                yv=torch.from_numpy(np.asarray(data['truth'][ids])).float().permute(0,1,4,2,3)
                xb=xv.to(a.device);yb=yv.to(a.device)
                if mean_x is not None:
                    xb=(xb-mean_x.to(a.device))/std_x.to(a.device)
                    yb=(yb-mean_y.to(a.device))/std_y.to(a.device)
                z=model.encoder(xb.reshape(len(ids),40,h,w));latent=integrate_batch(rhs,z,np.asarray(data['times'][ids]));pred=model(latent.flatten(0,1)).reshape(len(ids),20,2,h,w)
                if mean_y is not None:
                    pred=pred*std_y.to(a.device)+mean_y.to(a.device)
                    yb=yb*std_y.to(a.device)+mean_y.to(a.device)
                batch_val=(pred-yb).square().sum()
                val_total=batch_val if val_total is None else val_total+batch_val
                val_count+=yb.numel()
            val=float(val_total)/val_count
        train_loss=float(loss_total)/max(1,loss_batches)
        log.append({'epoch':epoch+1,'train_loss':train_loss,'val_loss':val,'learning_rate':opt.param_groups[0]['lr'],
                    'training_samples_seen':epoch_samples,'training_samples_total':len(train_idx),
                    'full_training_epoch':epoch_samples==len(train_idx),'validation_samples':len(val_idx)})
        if val < best:
            best=val
            best_epoch=epoch+1
            torch.save({'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'rhs':{k:v.detach().cpu().clone() for k,v in rhs.state_dict().items()},'optimizer':opt.state_dict(),'rhs_optimizer':rhs_opt.state_dict(),'lambda_teacher':a.lambda_teacher,'lambda_real':a.lambda_real,'architecture':a.architecture,'dim':a.dim,'width':a.width,'rhs_width':a.rhs_width,'equation_artifact':equation_artifact,'dynamics_kind':'symbolic' if equation_artifact else 'neural_warmup','best_val':best,'selection':'real validation MSE','shape':[h,w],'normalization':normalization,'test_targets_used_in_selection':False},out/'student.pt')
        if val < meaningful_best-a.min_delta:
            meaningful_best=val
            last_meaningful_epoch=epoch+1
        (out/'training.json').write_text(json.dumps({'status':'RUNNING','history':log,'best_val':best},indent=2))
        print('STUDENT_EPOCH',epoch+1,'train_loss',log[-1]['train_loss'],'val_mse',val,'best_val_mse',best,flush=True)
        if a.max_updates is not None and updates >= a.max_updates:
            break
        if a.patience and epoch+1-last_meaningful_epoch>=a.patience:
            break
    peak_gpu_gib=(torch.cuda.max_memory_allocated(a.device)/1024**3 if str(a.device).startswith('cuda') else None)
    (out/'training.json').write_text(json.dumps({'status':'COMPLETE','history':log,'best_val':best,'best_epoch':best_epoch,'last_meaningful_epoch':last_meaningful_epoch,'early_stopping_min_delta':a.min_delta,'epochs_run':len(log),'early_stopped':bool(a.patience and len(log)<a.epochs and (a.max_updates is None or updates<a.max_updates)),'updates':updates,'peak_gpu_allocated_gib':peak_gpu_gib},indent=2))


if __name__ == '__main__':
    main()
