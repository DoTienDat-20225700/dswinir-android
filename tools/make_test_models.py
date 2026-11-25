"""Identity-only instrumentation fixtures. NOT restoration models/checkpoints."""
from pathlib import Path
import tensorflow as tf
import numpy as np
root=Path('android/app/src/androidTest/assets')
root.mkdir(parents=True,exist_ok=True)
for name,shape in [('identity_nhwc',[1,256,256,3]),('identity_nchw',[1,3,256,256]),('wrong_size',[1,32,32,3])]:
    fn=tf.function(lambda x:x,input_signature=[tf.TensorSpec(shape,tf.float32)])
    model=tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()]).convert()
    (root/(name+'.tflite')).write_bytes(model)

# Artificial one-box COCO0 output. No learned weights or quality claim.
for segment in [False,True]:
    det=np.zeros((1,116 if segment else 84,1),np.float32)
    det[0,:4,0]=[.5,.5,.5,.5]; det[0,4,0]=.9
    if segment: det[0,84,0]=1
    proto=np.zeros((1,8,8,32),np.float32); proto[...,0]=1
    def outputs(x):
        # Keep a true runtime input dependency so the fixture has one input.
        zero=tf.reduce_sum(x)*1e-30
        d=tf.constant(det)+zero
        return (d,tf.constant(proto)+zero) if segment else d
    fn=tf.function(outputs,input_signature=[tf.TensorSpec([1,32,32,3],tf.float32)])
    model=tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()]).convert()
    (root/('yolo_seg_fixture.tflite' if segment else 'yolo_det_fixture.tflite')).write_bytes(model)
