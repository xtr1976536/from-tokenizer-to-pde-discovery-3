"""Evaluate a frozen student on the dataset's locked test windows."""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from distill_models import DistillStudent
from train_distill import integrate_batch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'config', 'student', 'freeze-record', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    root = Path(a.root).resolve()
    cfg = json.loads(Path(a.config).read_text())
    student_path = Path(a.student)
    checkpoint = torch.load(student_path, map_location='cpu', weights_only=False)
    student_hash = digest(student_path)
    frozen_path = Path(a.freeze_record)
    if a.split == 'test':
        frozen = json.loads(frozen_path.read_text())
        if frozen.get('test_targets_used_in_selection') is not False or frozen.get('student_sha256') != student_hash:
            raise ValueError('Freeze record must exclude test targets and match this student checkpoint')

    sys.path.insert(0, str(root / 'ast_bpe_multifield_v4'))
    from multitrajectory_data import RealWindows

    clean_path = root / cfg['data']['clean_protocol_manifest']
    reader = RealWindows(
        [Path(f) for pattern in cfg['data']['arrow_paths'] for f in sorted(glob.glob(str(root / pattern)))],
        root / cfg['data']['index_root'], require_full=True, spatial_stride=1,
        index_override={s: root / v for s, v in cfg['data']['index_override'].items()},
        clean_manifest=json.loads(clean_path.read_text()),
    )
    if reader.manifest['index_sha256'] != cfg['data']['effective_index_sha256']:
        raise ValueError('Dataset split indices differ from the frozen protocol')
    if a.split == 'test':
        reader.unlock_student_test(frozen_path, student_path)

    normalization = checkpoint.get('normalization')
    if not normalization:
        raise ValueError('Student checkpoint does not include training-only normalization statistics')
    mean_x = torch.tensor(normalization['mean_input'], dtype=torch.float32)
    mean_y = torch.tensor(normalization['mean_target'], dtype=torch.float32)
    std_x = torch.tensor(normalization['std_input'], dtype=torch.float32)
    std_y = torch.tensor(normalization['std_target'], dtype=torch.float32)
    device = torch.device(a.device)
    model = DistillStudent(*checkpoint['shape'], checkpoint['dim'], checkpoint['width']).to(device)
    model.load_state_dict(checkpoint['state'], strict=True)
    if checkpoint['dynamics_kind'] == 'symbolic':
        from symbolic_dynamics import SymbolicDynamics
        eq = checkpoint['equation_artifact']
        sys.path.insert(0, str(root / 'ast_bpe_multifield_v1'))
        from pipeline import parse_system
        system = parse_system(eq['text'], fields=tuple(eq['fields']), spatial=())
        rhs = SymbolicDynamics(system, eq['coefficients']).to(device)
        rhs.load_state_dict(checkpoint['rhs'], strict=True)
    else:
        from distill_models import LatentRHS
        rhs = LatentRHS(checkpoint['dim'], checkpoint['rhs_width']).to(device)
        rhs.load_state_dict(checkpoint['rhs'], strict=True)
    rhs.eval()
    model.eval()
    entries = sorted(reader.indices[a.split], key=lambda row: (row['sim_id'], row['time_id']))
    if a.limit:
        entries = entries[:a.limit]
    per_trajectory = {}
    for entry in entries:
        obs = reader.observed(a.split, entry)
        x = np.stack([obs['fields']['u'], obs['fields']['v']], -1).astype(np.float32)
        xx = torch.from_numpy(x)[None].to(device)
        xx = (xx - mean_x[:2].to(device)) / std_x[:2].clamp_min(1e-8).to(device)
        xx = xx.permute(0, 1, 4, 2, 3)
        dt = float(np.median(np.diff(obs['t'])))
        t = np.arange(21, dtype=np.float64) * dt
        with torch.inference_mode():
            z0 = model.encoder(xx.reshape(1, 40, *x.shape[1:3]))
            latent = integrate_batch(rhs, z0, t)
            pred = model(latent.reshape(-1, checkpoint['dim'])).reshape(1, 20, 2, *x.shape[1:3])
            pred = pred.permute(0, 1, 3, 4, 2)[0].cpu().numpy()
        pred = pred * std_y[:2].cpu().numpy().reshape(1, 1, 1, 2) + mean_y[:2].cpu().numpy().reshape(1, 1, 1, 2)
        y = reader.target(a.split, entry).astype(np.float32)
        err = (pred - y) ** 2
        row = per_trajectory.setdefault(entry['sim_id'], [0.0, 0])
        row[0] += float(err.sum())
        row[1] += int(err.size)
    report = {
        'status': 'COMPLETE', 'frozen_student': str(student_path.resolve()),
        'student_sha256': student_hash, 'freeze_record_sha256': digest(frozen_path) if a.split == 'test' else None, 'split': a.split,
        'test_targets_used_for_selection': False, 'windows': len(entries),
        'per_trajectory_rmse': {sid: (sse / n) ** 0.5 for sid, (sse, n) in per_trajectory.items()},
        'overall_rmse': (sum(v[0] for v in per_trajectory.values()) / sum(v[1] for v in per_trajectory.values())) ** 0.5,
        'access_log': reader.access_log,
    }
    Path(a.output).write_text(json.dumps(report, indent=2))
    print('TEST_EVALUATION_COMPLETE', report['windows'], report['overall_rmse'])


if __name__ == '__main__':
    main()
