"""Gate G2: run the exported ONNX with onnxruntime, compare to torch reference.

Usage:
    python3 convert/validate_onnx.py
Reads:
    convert/out/dswinir.onnx  ref_input.pt  ref_output.pt
Pass if max abs diff (ONNX vs torch) < 1e-3.
"""
import os
import sys

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
onnx_path = os.path.join(OUT, "dswinir.onnx")
if not os.path.exists(onnx_path):
    sys.exit(f"missing {onnx_path} — run convert/export.py first")

import onnxruntime as ort
import numpy as np

x = torch.load(os.path.join(OUT, "ref_input.pt")).numpy()
ref = torch.load(os.path.join(OUT, "ref_output.pt")).numpy()
print("input", x.shape, x.min(), x.max())
print("ref  ", ref.shape, ref.min(), ref.max())

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
o = sess.run(None, {"input": x})[0]
d = float(np.abs(o - ref).max())
print(f"ONNX out range {o.min():.4f}..{o.max():.4f}")
print(f"max abs diff (ONNX vs torch) = {d:.4e}")
assert d < 1e-3, "divergence too large — model conversion broken"
print("G2 ONNX GATE PASS")
