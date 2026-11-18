"""Stream seen train/val windows, decoding each Arrow trajectory once."""
import argparse,glob,json,sys,hashlib,collections
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser()
 for name in ('root','config','teacher','normalizer','output'):p.add_argument('--'+name,required=True)
 p.add_argument('--micro-batch',type=int,default=1);p.add_argument('--limit-per-split',type=int);a=p.parse_args();root=Path(a.root).resolve();cfg=json.loads(Path(a.config).read_text());out=Path(a.output)
 sys.path.insert(0,str(root/'ast_bpe_multifield_v4'))
 from multitrajectory_data import RealWindows,digest
 pa=None
 try:import pyarrow as pa;import pyarrow.ipc as ipc
 except ModuleNotFoundError:raise RuntimeError('pyarrow required to stream official Arrow trajectories')
 clean_path=Path(cfg['data']['clean_protocol_manifest']);
 if digest(clean_path)!=cfg['data']['clean_protocol_manifest_sha256']:raise ValueError('Clean manifest hash mismatch')
 arrows=[Path(f) for pattern in cfg['data']['arrow_paths'] for f in sorted(glob.glob(str(root/pattern)))]
 reader=RealWindows(arrows,root/cfg['data']['index_root'],require_full=True,spatial_stride=1,index_override={s:root/v for s,v in cfg['data']['index_override'].items()},clean_manifest=json.loads(clean_path.read_text()))
 if reader.manifest['index_sha256']!=cfg['data']['effective_index_sha256']:raise ValueError('Seen indices differ from protocol')
 dsroot=root/'data/baselines_seen_v1/cylinder';norm=torch.load(a.normalizer,map_location='cpu',weights_only=True)
 mean_x,mean_y,std_x,std_y=norm
 mean_x=mean_x.float().cuda().view(1,1,1,1,3);std_x=torch.where(std_x==0,1,std_x).float().cuda().view(1,1,1,1,3)
 mean_y=mean_y.float().cuda().view(1,1,1,1,3);std_y=torch.where(std_y==0,1,std_y).float().cuda().view(1,1,1,1,3)
 from realpdebench.model.load_model import load_model
 config_path=root/'baseline_configs/unet_cylinder_seen_v1.yaml';import yaml
 model_cfg=yaml.safe_load(config_path.read_text())
 class ShapeDataset:
  def __getitem__(self,i):return torch.zeros((20,64,128,3)),torch.zeros((20,64,128,3))
  def __len__(self):return 1
 model=load_model(ShapeDataset(),device='cpu',**model_cfg)
 ck=torch.load(a.teacher,map_location='cpu',weights_only=False)
 best=ck.get('best_iteration',ck.get('iteration'))
 if ck.get('iteration')!=best:raise ValueError('Teacher checkpoint is not the validation-best checkpoint')
 model.load_state_dict(ck['model_state_dict'],strict=True);model.cuda().eval()
 entries={s:reader.indices[s] for s in ('train','val')};groups=collections.defaultdict(list)
 if a.limit_per_split is None:
  expected={'train':cfg.get('train_windows'),'val':cfg.get('validation_windows')}
  if all(value is not None for value in expected.values()):
   actual={split:len(items) for split,items in entries.items()}
   if actual!=expected:raise ValueError(f'Full cache split counts differ from config: {actual} != {expected}')
 if a.limit_per_split is not None:entries={s:v[:a.limit_per_split] for s,v in entries.items()}
 for split,items in entries.items():
  for item in items:groups[item['sim_id']].append((split,item))
 total=sum(map(len,entries.values()));out.mkdir(parents=True,exist_ok=False)
 arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype='float32',shape=(total,20,64,128,2)) for k in ('x','teacher','truth')}
 times=np.lib.format.open_memmap(out/'times.npy',mode='w+',dtype='float64',shape=(total,21));splits=np.lib.format.open_memmap(out/'split.npy',mode='w+',dtype='<U5',shape=(total,));manifest_entries=[];offset=0
 for sid,items in groups.items():
  source=next(path for path,i in reader.rows[sid: sid+1]) if False else reader.rows[sid][0];row_index=reader.rows[sid][1]
  if str(source) not in reader._tables:
   import pyarrow.ipc as ipc
   with pa.memory_map(str(source),'r') as stream:reader._tables[str(source)]=ipc.open_stream(stream).read_all()
  r=reader._row(sid);ts=np.frombuffer(r['t'],np.float64)
  for begin in range(0,len(items),a.micro_batch):
   chunk=items[begin:begin+a.micro_batch];xs=[];ys=[];flips=[];grids=[]
   for split,item in chunk:
    start=int(item['time_id'])
    obs=reader.observed(split,item);fields=obs['fields'];x=np.stack([fields['u'][:20],fields['v'][:20]],-1)
    if split=='train':
     y=np.stack([fields['u'][20:40],fields['v'][20:40]],-1)
    else:
     y=reader.target(split,item)
    xs.append(x.astype(np.float32));ys.append(y.astype(np.float32));flips.append(obs['storage_flip_axes']);t=ts[start+19:start+40].copy();t-=t[0];grids.append(t)
   stored_inputs=[np.flip(x,axis=tuple(axes)).copy() if axes else x for x,axes in zip(xs,flips)]
   xb=torch.from_numpy(np.stack(stored_inputs)).float().cuda();xn=torch.cat([xb,torch.zeros_like(xb[...,:1])],-1);xn=(xn-mean_x)/std_x
   with torch.inference_mode():pred=model(xn)*std_y+mean_y
   for j,(split,item) in enumerate(chunk):
    teacher=pred[j,...,:2].cpu().numpy()
    if flips[j]:teacher=np.flip(teacher,axis=tuple(flips[j])).copy()
    arrays['x'][offset]=xs[j];arrays['truth'][offset]=ys[j];arrays['teacher'][offset]=teacher;times[offset]=grids[j];splits[offset]=split;manifest_entries.append(dict(split=split,**item));offset+=1
   if offset%100==0:print('CACHE',offset,'/',total,flush=True)
  reader._tables.pop(str(source),None);reader._cached_id=None;reader._cached_row=None
 for arr in list(arrays.values())+[times,splits]:arr.flush()
 digest_files={s:digest(root/v) for s,v in cfg['data']['index_override'].items()}
 normalization={k:v.float().reshape(-1)[:2].tolist() for k,v in zip(('mean_input','mean_target','std_input','std_target'),norm)}
 (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETE' if a.limit_per_split is None else 'PREFLIGHT_PARTIAL',teacher_checkpoint=str(Path(a.teacher).resolve()),teacher_iteration=best,normalizer_sha256=digest(a.normalizer),normalization=normalization,clean_manifest_sha256=digest(clean_path),effective_index_sha256=digest_files,entries=manifest_entries,test_read=False,train_windows=len(entries['train']),validation_windows=len(entries['val']),decoded_trajectories=len(groups),micro_batch=a.micro_batch),indent=2));print('CACHE_COMPLETE',offset,flush=True)
if __name__=='__main__':main()
