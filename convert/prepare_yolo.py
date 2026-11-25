"""YOLO11 FP32 ONNX export, raw normalized TFLite xywh, with Torch fixtures.
Run with PYTHONPATH=.yolo-deps:.export-deps and ultralytics==8.3.221.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True); p.add_argument('--size',type=int,default=640)
a=p.parse_args()
if ultralytics.__version__!='8.3.221': raise RuntimeError('Use pinned ultralytics==8.3.221')
root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
model=YOLO(a.checkpoint).model.cpu().float().eval()
for layer in model.modules():
    if isinstance(layer,Detect):
        layer.export=True; layer.format='tflite'; layer.dynamic=False
rng=np.random.default_rng(27)
inputs=[rng.random((1,3,a.size,a.size),dtype=np.float32),np.broadcast_to(np.linspace(0,1,a.size,dtype=np.float32)[None,None,:,None],(1,3,a.size,a.size)).copy()]
outputs=None
with torch.inference_mode():
    for i,x in enumerate(inputs):
        y=model(torch.from_numpy(x)); outputs=list(y) if isinstance(y,(list,tuple)) else [y]
        np.save(root/f'input_{i}.npy',x)
        for j,t in enumerate(outputs): np.save(root/f'reference_{i}_{j}.npy',t.cpu().numpy())
        print('Torch fixture',i,[list(t.shape) for t in outputs],flush=True)
    torch.onnx.export(model,torch.from_numpy(inputs[0]),str(root/'model.onnx'),opset_version=13,input_names=['images'],output_names=['detections','prototypes'][:len(outputs)],do_constant_folding=True)
metadata=dict(checkpoint=a.checkpoint,sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),ultralytics=ultralytics.__version__,torch=torch.__version__,size=a.size,outputs=len(outputs),format='normalized xywh, raw outputs, nms=False, COCO80')
(root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved',root/'model.onnx',flush=True)
