"""Prepare FP32 weights and deterministic reference fixtures in PyTorch.
Run with a NumPy compatible with your PyTorch (this workspace: PYTHONPATH=.export-deps).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate_model import sd_of,infer_config
from model import MSDeformableNAIR

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--out',default='convert/prepared')
p.add_argument('--size',type=int,default=256,choices=[128,256])
p.add_argument('--device',default='cpu',choices=['cpu','cuda','auto'])
a=p.parse_args()
root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
sd=sd_of(a.checkpoint); cfg=infer_config(sd)
model=MSDeformableNAIR(**cfg); model.load_state_dict(sd,strict=True); model.eval()
device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
model.to(device)
torch.set_num_threads(4)
np.savez(root/'weights.npz',**{k:v.detach().numpy() for k,v in sd.items()})
meta={'config':cfg,'checkpoint':a.checkpoint,'sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
      'size':a.size,'reference_device':device,'global_residual':True,'note':'Default flags from official source; task coverage must be verified from checkpoint provenance.'}
(root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
# Structured, deterministic inputs exercise interpolation and image borders.
rng=np.random.default_rng(42)
inputs=[rng.random((1,3,a.size,a.size),dtype=np.float32),
        np.broadcast_to(np.linspace(0,1,a.size,dtype=np.float32)[None,None,None,:],(1,3,a.size,a.size)).copy()]
for i,x in enumerate(inputs):
    with torch.inference_mode(): y=model(torch.from_numpy(x).to(device)).cpu().numpy()
    if not np.isfinite(y).all(): raise RuntimeError('Non-finite reference')
    np.save(root/f'input_{i}.npy',x.transpose(0,2,3,1))
    np.save(root/f'reference_{i}.npy',y.transpose(0,2,3,1))
    print('Reference',i,'range',float(y.min()),float(y.max()),flush=True)
print('Prepared',root,flush=True)
