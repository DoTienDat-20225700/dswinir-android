"""Stage real model/reference fixtures for optional Android instrumentation."""
import json,os,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/device-parity'; OUT.mkdir(parents=True,exist_ok=True)
cases=[]
for task,folder,filename in [('restoration','convert/prepared256','dswinir_256_fp32.tflite'),('detection','convert/yolo/detection','yolo11n_fp32.tflite'),('segmentation','convert/yolo/segmentation','yolo11n-seg_fp32.tflite')]:
 source=ROOT/folder/filename
 if not source.is_file(): raise FileNotFoundError(source)
 target=OUT/filename
 if not target.exists():
  try: os.link(source,target)
  except OSError: shutil.copyfile(source,target)
 for i in range(2):
  x=np.load(ROOT/folder/f'input_{i}.npy')
  if task!='restoration': x=x.transpose(0,2,3,1)
  input_name=f'{task}-{i}-input.bin'; x.astype('<f4').tofile(OUT/input_name)
  refs=[]
  count=2 if task=='segmentation' else 1
  for j in range(count):
   name=f'reference_{i}.npy' if task=='restoration' else f'reference_{i}_{j}.npy'
   y=np.load(ROOT/folder/name)
   if task=='segmentation' and y.ndim==4: y=y.transpose(0,2,3,1)
   output_name=f'{task}-{i}-output-{j}.bin'; y.astype('<f4').tofile(OUT/output_name)
   refs.append(dict(file=output_name,shape=list(y.shape)))
  cases.append(dict(name=f'{task}-{i}',model=filename,input=input_name,outputs=refs))
(OUT/'manifest.json').write_text(json.dumps(dict(cases=cases),indent=2)+'\n')
print(OUT)
