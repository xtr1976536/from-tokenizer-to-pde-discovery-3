"""Recover the exact train/val window entries retained in the cache manifest.
The test index is not reconstructable from this training cache.
"""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'data_protocol/cache_manifest.json').read_text())
a.output.mkdir(parents=True,exist_ok=False)
for split in ('train','val'):
    entries=[{k:v for k,v in row.items() if k!='split'} for row in m['entries'] if row['split']==split]
    (a.output/f'{split}_index_real.json').write_text(json.dumps(entries,indent=2))
    print(split,len(entries))
print('Recovered semantic window entries; original file serialization hashes are not reproduced.')
