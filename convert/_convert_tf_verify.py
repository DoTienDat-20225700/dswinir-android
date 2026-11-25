"""ONNX -> TFLite (via onnx2tf) + numeric verification.  Run under the VENV:

    venv/bin/python convert/_convert_tf_verify.py [size]

Expects in convert/out/:
    dswinir_<size>.onnx  dswinir_<size>_input.npy  dswinir_<size>_ref.npy
(produced by convert/_export_bridge.py under system python/torch).

Steps:
  1. onnx.checker on the .onnx.
  2. onnxruntime: run the .onnx on input.npy, compare vs ref.npy.
  3. onnx2tf CLI: build dswinir_<size>.tflite.
  4. tf.lite: run the .tflite on the same input, compare vs ref.npy.
Prints max_abs_err and rel L2 for each stage.
"""
import os
import subprocess
import sys

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"


def cmp(ref, got, name):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    max_abs = float(np.max(np.abs(ref - got)))
    denom = float(np.linalg.norm(ref))
    rel = float(np.linalg.norm(ref - got) / (denom + 1e-12))
    print(f"[{name}] shape={ref.shape} max_abs_err={max_abs:.6g}  rel_L2={rel:.6g}", flush=True)
    return max_abs, rel


def main(size):
    onnx_path = os.path.join(OUT, f"dswinir_{size}.onnx")
    input_path = os.path.join(OUT, f"dswinir_{size}_input.npy")
    ref_path = os.path.join(OUT, f"dswinir_{size}_ref.npy")
    for p in (onnx_path, input_path, ref_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - run convert/_export_bridge.py first (system python)")
    x = np.load(input_path)
    ref = np.load(ref_path)
    print(f"input {x.shape} ref {ref.shape}", flush=True)

    # 1. onnx checker
    import onnx
    onnx.checker.check_model(onnx_path)
    print(f"[onnx] checker OK {os.path.getsize(onnx_path)//1024//1024} MB", flush=True)

    # 2. onnxruntime numeric
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]
    cmp(ref, ort_out, "onnxruntime")

    # 3. onnx2tf -> tflite.  onnx2tf 1.16: -o is an output FOLDER; it writes
    #     <folder>/<onnx_basename>.tflite (same base name as the input .onnx).
    tflite_path = os.path.join(OUT, f"dswinir_{size}.tflite")
    tfmodel_dir = os.path.join(OUT, f"dswinir_{size}_tfmodel")
    os.makedirs(tfmodel_dir, exist_ok=True)
    onnx2tf_out = os.path.join(tfmodel_dir, f"dswinir_{size}.tflite")
    print("running onnx2tf ...", flush=True)
    subprocess.run([sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", tfmodel_dir, "-n"],
                   check=True)
    if not os.path.exists(onnx2tf_out):
        raise SystemExit(f"onnx2tf did not produce {onnx2tf_out}")
    # normalize to the expected path so the rest of this script is size-stable
    os.replace(onnx2tf_out, tflite_path)
    print(f"[onnx2tf] wrote {tflite_path} {os.path.getsize(tflite_path)//1024//1024} MB", flush=True)

    # 4. tf.lite numeric
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    inp = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    interp.allocate_tensors()
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    lite_out = interp.get_tensor(outd["index"])
    cmp(ref, lite_out, "tflite")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 128)
