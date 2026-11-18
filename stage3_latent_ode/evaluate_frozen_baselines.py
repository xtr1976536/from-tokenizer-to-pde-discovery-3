"""Evaluate frozen seen U-Net and FNO on the same windows as both students."""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluate_frozen_student_test import digest


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'config', 'unet', 'fno', 'normalizer', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--full-result')
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    root = Path(a.root).resolve()
    cfg = json.loads(Path(a.config).read_text())
    sys.path[:0] = [str(root / 'ast_bpe_multifield_v4'), str(root / 'research/realpdebench/upstream')]
    from multitrajectory_data import RealWindows
    from realpdebench.model.load_model import load_model

    reader = RealWindows(
        [Path(path) for pattern in cfg['data']['arrow_paths'] for path in sorted(glob.glob(str(root / pattern)))],
        root / cfg['data']['index_root'], require_full=True, spatial_stride=1,
        index_override={split: root / path for split, path in cfg['data']['index_override'].items()},
        clean_manifest=json.loads((root / cfg['data']['clean_protocol_manifest']).read_text()),
    )
    if reader.manifest['index_sha256'] != cfg['data']['effective_index_sha256']:
        raise ValueError('Seen split index hash mismatch')
    if a.split == 'test':
        if not a.full_result:
            raise ValueError('Frozen two-route result is required before test evaluation')
        result = json.loads(Path(a.full_result).read_text())
        if result['status'] != 'COMPLETE' or set(result['routes']) != set(cfg['routes']):
            raise ValueError('Both student routes must be complete before test evaluation')
        for route in result['routes'].values():
            freeze = json.loads(Path(route['freeze_record']).read_text())
            if freeze['student_sha256'] != digest(route['student']) or freeze['test_targets_used_in_selection'] is not False:
                raise ValueError('Student freeze record mismatch')
        first = next(iter(result['routes'].values()))
        reader.unlock_student_test(first['freeze_record'], first['student'])

    mean_x, mean_y, std_x, std_y = torch.load(a.normalizer, map_location='cpu', weights_only=True)
    device = torch.device(a.device)
    mean_x, mean_y = mean_x.to(device), mean_y.to(device)
    std_x = torch.where(std_x == 0, 1, std_x).to(device)
    std_y = torch.where(std_y == 0, 1, std_y).to(device)

    class ShapeDataset:
        def __getitem__(self, index):
            return torch.zeros((20, 64, 128, 3)), torch.zeros((20, 64, 128, 3))

        def __len__(self):
            return 1

    models = {}
    for name, path in (('unet', a.unet), ('fno', a.fno)):
        settings = yaml.safe_load((root / f'baseline_configs/{name}_cylinder_seen_v1.yaml').read_text())
        model = load_model(ShapeDataset(), device='cpu', **settings)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['iteration'] != checkpoint['best_iteration']:
            raise ValueError(f'{name} checkpoint is not its validation-best iteration')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        models[name] = model.to(device).eval()

    entries = sorted(reader.indices[a.split], key=lambda row: (row['sim_id'], row['time_id']))
    if a.limit:
        entries = entries[:a.limit]
    totals = {name: {} for name in models}
    for number, entry in enumerate(entries, 1):
        obs = reader.observed(a.split, entry)
        x = np.stack([obs['fields']['u'], obs['fields']['v'], np.zeros_like(obs['fields']['u'])], -1).astype('float32')
        for axis in obs['storage_flip_axes']:
            x = np.flip(x, axis=axis)
        tensor = torch.from_numpy(x.copy())[None].to(device)
        normalized = (tensor - mean_x) / std_x
        with torch.inference_mode():
            preds = {name: (model(normalized)[0] * std_y + mean_y)[..., :2].cpu().numpy()
                     for name, model in models.items()}
        truth = reader.target(a.split, entry)
        for name, prediction in preds.items():
            for axis in obs['storage_flip_axes']:
                prediction = np.flip(prediction, axis=axis)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise ValueError(f'{name} prediction is invalid')
            row = totals[name].setdefault(entry['sim_id'], [0.0, 0])
            row[0] += float(np.square(prediction - truth).sum())
            row[1] += int(truth.size)
        if number % 100 == 0:
            print('BASELINE_EVALUATED', number, '/', len(entries), flush=True)
    report = {'status': 'COMPLETE', 'split': a.split, 'windows': len(entries),
              'checkpoint_sha256': {'unet': digest(a.unet), 'fno': digest(a.fno)},
              'models': {name: {'overall_rmse': (sum(v[0] for v in rows.values()) /
                                                sum(v[1] for v in rows.values())) ** 0.5,
                                'per_trajectory_rmse': {sid: (sse / count) ** 0.5
                                                        for sid, (sse, count) in rows.items()}}
                         for name, rows in totals.items()}}
    Path(a.output).write_text(json.dumps(report, indent=2))
    print('BASELINE_EVALUATION_COMPLETE', a.split, len(entries))


if __name__ == '__main__':
    main()
