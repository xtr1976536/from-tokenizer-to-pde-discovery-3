"""Real-only, replayable official window reader. Never downloads raw datasets.

Training sees the complete indexed 20+20 frames. Validation observations expose
20 input frames; validation targets are allowed for model selection. Test target
access is locked until a frozen model artifact has been recorded.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'ast_bpe_multifield_v1'))
from grid import GridBackend
from protocol_checks import frame_audit,verify_provenance,full_window_audit,train_reward_partition


def arrow_modules():
    # Prefer the host environment (especially Linux/CUDA). The existing local
    # fallback contains macOS CPython 3.12 wheels and must not shadow server pip.
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ModuleNotFoundError:
        if sys.platform!='darwin' or sys.version_info[:2]!=(3,12):
            raise RuntimeError('Install pyarrow in the active server Python environment')
        sys.path.append(str(ROOT.parent/'ast_bpe_multifield_v2/.deps'))
        import pyarrow as pa
        import pyarrow.ipc as ipc
    return pa,ipc


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def observation(t,x,y,fields,spatial_stride=1,fit_stride=2):
    # Preserve the real physical coordinate ordering, including descending axes.
    delta=np.diff(t)
    if len(delta)<2 or not np.allclose(delta,np.median(delta),rtol=1e-6,atol=1e-10):
        raise ValueError('Fixed-step Cylinder forecast requires uniformly spaced physical timestamps')
    coords={'y':y[:,0].copy(),'x':x[0].copy()} if np.ptp(x[0])>np.ptp(x[:,0]) else {'x':x[:,0].copy(),'y':y[0].copy()}
    if tuple(coords)==('y','x') and (not np.allclose(x,x[0][None,:]) or not np.allclose(y,y[:,0][:,None])):
        raise ValueError('Curvilinear coordinates cannot be silently treated as a Cartesian grid')
    fields={f:np.asarray(a,dtype=float)[:,::spatial_stride,::spatial_stride].copy() for f,a in fields.items()}
    coords={k:a[::spatial_stride] for k,a in coords.items()}
    flips=[]
    for axis,(name,values) in enumerate(coords.items(),1):
        if values[1]<values[0]:
            flips.append(axis)
            coords[name]=values[::-1].copy();fields={f:np.flip(a,axis=axis).copy() for f,a in fields.items()}
    shape=fields['u'].shape
    if min(shape)<5:raise ValueError('Grid too small for second-order derivative fitting')
    mask=np.zeros(shape,bool);mask[2:-2:fit_stride,2:-2:fit_stride,2:-2:fit_stride]=True
    backend=GridBackend({'t':t,**coords},fields)
    tr=backend.trajectory(('u','v'),mask,np.zeros_like(mask));tr['spatial']=tuple(coords)
    tr['raw_fields']=fields
    tr['raw_coordinates']=coords
    tr['raw_t']=np.asarray(t)
    return dict(t=np.asarray(t),coordinates=coords,fields=fields,trajectory=tr,dt=float(np.median(np.diff(t))),storage_flip_axes=flips)


def storage_orientation(array,obs):
    """Undo physical-axis flips before official first-octant Fourier metrics."""
    for axis in obs['storage_flip_axes']:array=np.flip(array,axis=axis)
    return array.copy()


class RealWindows:
    def __init__(self,arrow_paths,index_root,require_full=False,trajectory_ids=None,
                 windows_per_trajectory=None,spatial_stride=1,fit_stride=2,index_override=None,clean_manifest=None,
                 search_windows_per_trajectory=None,search_spatial_stride=4,search_fit_stride=2):
        pa,ipc=arrow_modules()
        self.index_root=Path(index_root);self.rows={};self.lengths={};self._cached_id=None;self._cached_row=None;self._tables={}
        self.test_frozen=None;self.access_log=[];self.spatial_stride=spatial_stride;self.fit_stride=fit_stride
        self.search_windows_per_trajectory=search_windows_per_trajectory
        self.search_spatial_stride=search_spatial_stride;self.search_fit_stride=search_fit_stride
        for path in map(Path,arrow_paths):
            with pa.memory_map(str(path),'r') as src:
                table=ipc.open_stream(src).read_all()
                for i in range(table.num_rows):
                    sid=table['sim_id'][i].as_py()
                    if sid in self.rows:raise ValueError('Duplicate real trajectory '+sid)
                    self.rows[sid]=(path,i)
                    self.lengths[sid]=table['shape_t'][i].as_py()
        if not self.rows:raise ValueError('No real Arrow trajectories')
        official_indices={s:json.loads((self.index_root/f'{s}_index_real.json').read_text()) for s in ('train','val','test')}
        all_indices=official_indices
        if index_override:
            if set(index_override)!={'train','val','test'}:raise ValueError('Provide all three explicit effective index paths')
            all_indices={s:json.loads(Path(index_override[s]).read_text()) for s in official_indices}
            if clean_manifest:
                if not full_window_audit(all_indices,clean_manifest['development_trajectories'])['full_window_isolated']:
                    raise ValueError('Unified clean protocol has full-window or development-set overlap')
            elif all_indices['test']!=official_indices['test']:raise ValueError('Adjusted profile must preserve the complete official test index')
            for split,rows in all_indices.items():
                allowed={(r['sim_id'],r['time_id']) for r in official_indices[split]}
                if any((r['sim_id'],r['time_id']) not in allowed for r in rows):raise ValueError('Adjusted indices must be subsets of released indices')
            if not frame_audit(all_indices)['stored_frame_isolated']:raise ValueError('Adjusted index profile still exposes held-out frames')
        needed={r['sim_id'] for entries in official_indices.values() for r in entries}
        missing=sorted(needed-self.rows.keys())
        if require_full and missing:raise ValueError(f'Full official evaluation missing {len(missing)} trajectories: {missing[:5]}')
        if require_full and (trajectory_ids or windows_per_trajectory or spatial_stride!=1):
            raise ValueError('Full official profile forbids trajectory/window caps or spatial subsampling')
        chosen=set(trajectory_ids) if trajectory_ids else set(self.rows)
        if chosen-self.rows.keys():raise ValueError('Requested real trajectories not available')
        self.indices={};self.entry_keys={};self.sorted_indices={}
        for split,entries in all_indices.items():
            selected=[dict(r) for r in entries if r['sim_id'] in chosen]
            if windows_per_trajectory:
                counts={};limited=[]
                for r in selected:
                    sid=r['sim_id']
                    if counts.get(sid,0)<windows_per_trajectory:
                        limited.append(r);counts[sid]=counts.get(sid,0)+1
                selected=limited
            if not selected:raise ValueError('Empty '+split+' split')
            if any(r['time_id']<0 or r['time_id']+40>self.lengths[r['sim_id']] for r in selected):
                raise ValueError('Official indexed horizon exceeds stored trajectory in '+split)
            if len({(r['sim_id'],r['time_id']) for r in selected})!=len(selected):raise ValueError('Duplicate split window')
            self.indices[split]=selected
            self.entry_keys[split]={(r['sim_id'],r['time_id']) for r in selected}
            self.sorted_indices[split]=sorted(selected,key=lambda r:(r['sim_id'],r['time_id']))
        if len({r['sim_id'] for r in self.indices['train']})<2:
            raise ValueError('Multitrajectory training requires at least two real training trajectories')
        self.manifest=dict(branch='real',task='Real-world Training',protocol='clean_real_v1' if clean_manifest else 'stored_frame_isolated40' if index_override else 'official',
            full_official=require_full and not bool(index_override),full_coverage=require_full,
            input_frames=20,output_frames=20,training_frames_per_window=40,spatial_stride=spatial_stride,
            fit_stride=fit_stride,extra_parameter_inputs=False,missing_official_trajectories=missing,
            official_index_sha256={s:digest(self.index_root/f'{s}_index_real.json') for s in self.indices},
            index_sha256={s:digest(index_override[s] if index_override else self.index_root/f'{s}_index_real.json') for s in self.indices},
            splits={s:dict(windows=len(v),trajectories=len({r['sim_id'] for r in v}),entries=v) for s,v in self.indices.items()},
            sources=[dict(path=str(p.resolve()),bytes=p.stat().st_size) for p in sorted(set(p for p,_ in self.rows.values()))],
            normalization='trajectory surrogate scales from real training observations only; dynamics/metrics in physical units',
            scope=('complete adjusted 40-frame-isolated real indices' if index_override else 'full released official real indices') if require_full else 'capped real multitrajectory integration test; not full benchmark')
        self.clean_manifest=clean_manifest
        if clean_manifest:self.manifest['clean_protocol']=clean_manifest

    def _row(self,sid):
        if self._cached_id==sid and self._cached_row is not None:
            return self._cached_row
        path,i=self.rows[sid]
        path_key=str(path)
        if path_key not in self._tables:
            pa,ipc=arrow_modules()
            with pa.memory_map(path_key,'r') as src:
                self._tables[path_key]=ipc.open_stream(src).read_all()
        table=self._tables[path_key]
        needed=('sim_id','shape_t','shape_h','shape_w','u','v','x','x_shape_h','x_shape_w','y','y_shape_h','y_shape_w','t')
        self._cached_row={name:table[name][i].as_py() for name in needed}
        self._cached_id=sid
        return self._cached_row

    def observed(self,split,entry):
        if split not in self.indices or (entry['sim_id'],entry['time_id']) not in self.entry_keys[split]:raise ValueError('Window not in selected official split')
        row=self._row(entry['sim_id']);shape=tuple(row[k] for k in ('shape_t','shape_h','shape_w'))
        start=int(entry['time_id']);count=40 if split=='train' else 20
        if start<0 or start+40>shape[0]:raise ValueError('Indexed forecast horizon exceeds trajectory')
        fields={f:np.frombuffer(row[f],np.float32).reshape(shape)[start:start+count] for f in ('u','v')}
        x=np.frombuffer(row['x'],np.float64).reshape(row['x_shape_h'],row['x_shape_w'])
        y=np.frombuffer(row['y'],np.float64).reshape(row['y_shape_h'],row['y_shape_w'])
        t=np.frombuffer(row['t'],np.float64)[start:start+count]
        obs=observation(t,x,y,fields,self.spatial_stride,self.fit_stride)
        if tuple(obs['coordinates'])!=('y','x'):raise ValueError('Cylinder joint PINN requires stored grid axes (y,x); transpose explicitly before use')
        obs.update(trajectory_id=entry['sim_id'],window=dict(entry),split=split)
        obs['trajectory']['trajectory_id']=entry['sim_id']
        return obs

    def windows(self,split):
        # Trajectory sorting keeps the single-row cache efficient. Index membership
        # and sample weights are unchanged; training shuffling happens at PINN level.
        for entry in self.sorted_indices[split]:
            yield self.observed(split,entry)

    def factory(self,split):
        def replay():
            for obs in self.windows(split):yield {'trajectory_id':obs['trajectory_id']},obs['trajectory']
        return replay

    def search_partition(self):
        if not hasattr(self,'_search_partition'):
            self._search_partition=train_reward_partition(self.indices['train'],self.search_windows_per_trajectory)
        return self._search_partition

    def search_factory(self,role):
        entries=self.search_partition()[role]
        def replay():
            for entry in entries:
                old_stride,old_fit=self.spatial_stride,self.fit_stride
                self.spatial_stride,self.fit_stride=self.search_spatial_stride,self.search_fit_stride
                try:obs=self.observed('train',entry)
                finally:self.spatial_stride,self.fit_stride=old_stride,old_fit
                yield {'trajectory_id':obs['trajectory_id']},obs['trajectory']
        return replay

    def unlock_test(self,frozen_artifact):
        path=Path(frozen_artifact)
        if not path.is_file():raise ValueError('Freeze selected coefficients before opening test targets')
        selected=json.loads(path.read_text())
        if not selected.get('text') or set(selected.get('coefficients',{}))!={'u','v'}:
            raise ValueError('Frozen artifact must contain the selected PDE system and coefficients')
        if selected.get('test_targets_used_in_selection') is not False:raise ValueError('Selection must explicitly exclude test targets')
        if not all(np.isfinite(c).all() for c in selected['coefficients'].values()):raise ValueError('Nonfinite frozen coefficients')
        if selected.get('source_sha256'):verify_provenance(selected['source_sha256'])
        self.test_frozen=dict(path=str(path.resolve()),sha256=digest(path))

    def unlock_student_test(self,frozen_artifact,student_checkpoint):
        path=Path(frozen_artifact)
        selected=json.loads(path.read_text())
        if selected.get('test_targets_used_in_selection') is not False:
            raise ValueError('Student selection must exclude test targets')
        if selected.get('student_sha256')!=digest(student_checkpoint):
            raise ValueError('Frozen student checksum mismatch')
        self.test_frozen=dict(path=str(path.resolve()),sha256=digest(path))

    def target(self,split,entry):
        if split not in ('val','test'):raise ValueError('Targets are for validation/evaluation only')
        if split=='test' and self.test_frozen is None:raise RuntimeError('Test future locked until model selection is frozen')
        if (entry['sim_id'],entry['time_id']) not in self.entry_keys[split]:raise ValueError('Target window outside selected official split')
        if split=='test' and digest(self.test_frozen['path'])!=self.test_frozen['sha256']:
            raise RuntimeError('Frozen selection artifact was modified during test')
        row=self._row(entry['sim_id']);shape=tuple(row[k] for k in ('shape_t','shape_h','shape_w'));start=int(entry['time_id'])+20
        fields={f:np.frombuffer(row[f],np.float32).reshape(shape)[start:start+20] for f in ('u','v')}
        x=np.frombuffer(row['x'],np.float64).reshape(row['x_shape_h'],row['x_shape_w'])
        y=np.frombuffer(row['y'],np.float64).reshape(row['y_shape_h'],row['y_shape_w'])
        t=np.frombuffer(row['t'],np.float64)[start:start+20]
        obs=observation(t,x,y,fields,self.spatial_stride,self.fit_stride)
        self.access_log.append(dict(split=split,window=dict(entry),purpose='forecast selection' if split=='val' else 'frozen-model evaluation'))
        return np.stack([obs['fields'][f] for f in ('u','v')],axis=-1)
