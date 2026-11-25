"""ONNX -> TF -> FP32 TFLite with canonical outputs and numerical gates.
Use workspace TF venv; no calibration dataset download or quantization.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnx
import tensorflow as tf
import onnx2tf

p=argparse.ArgumentParser(); p.add_argument('--prepared',required=True); a=p.parse_args()
root=Path(a.prepared); meta=json.loads((root/'metadata.json').read_text())
inputs=[np.load(root/f'input_{i}.npy') for i in range(2)]
refs=[[np.load(root/f'reference_{i}_{j}.npy') for j in range(meta['outputs'])] for i in range(2)]
def compare(ref,got,name):
 if ref.shape!=got.shape or not np.isfinite(got).all(): raise RuntimeError(f'{name}: shape/nonfinite mismatch {ref.shape} {got.shape}')
 error=float(np.max(np.abs(ref-got))); print(name,'max_abs',error,flush=True)
 if error>1e-3: raise RuntimeError(f'{name}: parity gate failed')
 return error
sess=ort.InferenceSession(str(root/'model.onnx'),providers=['CPUExecutionProvider'])
onnx_errors=[]
for i,x in enumerate(inputs):
 ys=sess.run(None,{'images':x}); onnx_errors.append([compare(r,y,f'ONNX {i}/{j}') for j,(r,y) in enumerate(zip(refs[i],ys))])
# onnx2tf 1.16 downloads sample photos even for FP32 unless replaced. Supply
# deterministic local fixtures for its internal shape/axis validation instead.
onnx2tf.onnx2tf.download_test_image_data=lambda: inputs[0][:,:,:128,:128].transpose(0,2,3,1)
# Older onnx2tf maps node names directly to Keras root scopes. Torch exports
# leading '/' names, which Keras rejects. Rename identifiers, preserving data.
graph=onnx.load(str(root/'model.onnx'))
def safe(name): return re.sub(r'[^A-Za-z0-9_.\\/>-]','_',name.lstrip('/'))
for node in graph.graph.node:
 node.name=safe(node.name)
 for j,name in enumerate(node.input): node.input[j]=safe(name)
 for j,name in enumerate(node.output): node.output[j]=safe(name)
for collection in [graph.graph.initializer,graph.graph.input,graph.graph.output,graph.graph.value_info]:
 for tensor in collection: tensor.name=safe(tensor.name)
onnx.checker.check_model(graph)
normalized=root/'model_tf_names.onnx'; onnx.save(graph,str(normalized))
model=onnx2tf.convert(input_onnx_file_path=str(normalized),output_folder_path=str(root/'tf'),disable_model_save=True,non_verbose=True)
outputs=[]
for t in model.outputs:
 shape=list(t.shape)
 if len(shape)==3:
  if shape[1] in [84,116]: outputs.insert(0,t)
  elif shape[2] in [84,116]: outputs.insert(0,tf.transpose(t,[0,2,1]))
  else: raise RuntimeError(f'Unexpected detection shape {shape}')
 elif len(shape)==4:
  if shape[-1]==32: outputs.append(t)
  elif shape[1]==32: outputs.append(tf.transpose(t,[0,2,3,1]))
  else: raise RuntimeError(f'Unexpected prototype shape {shape}')
canonical=tf.keras.Model(model.inputs,outputs)
converter=tf.lite.TFLiteConverter.from_keras_model(canonical)
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]; converter.optimizations=[]
data=converter.convert()
lite=tf.lite.Interpreter(model_content=data,num_threads=4); lite.allocate_tensors()
inp=lite.get_input_details()[0]; outs=lite.get_output_details()
errors=[]
for i,x in enumerate(inputs):
 lite.set_tensor(inp['index'],x.transpose(0,2,3,1)); lite.invoke()
 got=[lite.get_tensor(t['index']) for t in outs]
 got.sort(key=lambda t:t.ndim)
 record=[]
 for j,(ref,y) in enumerate(zip(refs[i],got)):
  if y.ndim==4: y=y.transpose(0,3,1,2)
  record.append(compare(ref,y,f'TFLite {i}/{j}'))
 errors.append(record)
name='yolo11n-seg_fp32.tflite' if meta['outputs']==2 else 'yolo11n_fp32.tflite'
(root/name).write_bytes(data)
(root/'conversion-report.json').write_text(json.dumps(dict(checkpoint=meta,onnx_max_abs=onnx_errors,tflite_max_abs=errors,tolerance=1e-3,tf_version=tf.__version__,tflite_sha256=hashlib.sha256(data).hexdigest(),model_bytes=len(data),quality_mAP='NOT_EVALUATED'),indent=2)+'\n')
print('Saved',root/name,'bytes',len(data),flush=True)
