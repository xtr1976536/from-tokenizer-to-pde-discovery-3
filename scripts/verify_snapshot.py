"""Verify release hashes and run each compact student without data or API access."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'stage3_latent_ode'),str(root/'stage3_neural_pde'),str(root/'ast_bpe_multifield_v1')]
from run_llm_student import load_student
from pipeline import parse_system
from symbolic_dynamics import SymbolicDynamics
from train_distill import integrate_batch

def verify():
    manifest=json.loads((root/'provenance_manifest.json').read_text())
    for row in manifest['files']:
        path=root/row['path']
        if row['path'].startswith('release_assets/') and not path.exists():
            print('OPTIONAL_RELEASE_ASSET_NOT_DOWNLOADED',row['path'])
            continue
        assert path.is_file(),row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
    torch.set_num_threads(2)
    torch.manual_seed(525)
    for row in json.loads((root/'results/snapshot_summary.json').read_text()):
        model,ck=load_student(root/row['best_checkpoint'],'cpu')
        assert ck['dim']==row['dim']
        assert ck['best_val']==row['best_validation_mse']
        eq=ck['equation_artifact']
        rhs=SymbolicDynamics(parse_system(eq['text'],fields=tuple(eq['fields']),spatial=()),eq['coefficients'])
        rhs.load_state_dict(ck['rhs'])
        with torch.inference_mode():
            z=model.encoder(torch.zeros(1,40,*ck['shape']))
            states=integrate_batch(rhs,z,np.linspace(0,0.01,21))
            prediction=model(states.flatten(0,1))
        assert prediction.shape==(20,2,*ck['shape'])
        assert torch.isfinite(prediction).all()
        print('CHECKPOINT_FORWARD_PASS',row['dim'],row['completed_epochs'],row['best_validation_mse'])
    print('SNAPSHOT_VERIFIED')

if __name__=='__main__':verify()
