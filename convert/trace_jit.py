"""Trace the rebuilt model to TorchScript for a later tf2onnx conversion.

JIT tracing only records a graph (one real forward pass) — it does NOT run
the ONNX symbolic exporters that OOMed on this low-RAM machine. tf2onnx in
the clean venv then converts the .pt to ONNX.

Usage:
    python3 convert/trace_jit.py [ckpt] [size]
Output:
    convert/out/dswinir_jit_{size}.pt
"""
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_model import sd_of, infer_config
from model import MSDeformableNAIR

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main(ckpt_path, size):
    sd = sd_of(ckpt_path)
    cfg = infer_config(sd)
    print("config:", cfg, flush=True)
    model = MSDeformableNAIR(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (list(missing)[:5], list(unexpected)[:5])
    del sd
    gc.collect()
    model.eval()

    x = torch.rand(1, 3, size, size, dtype=torch.float32)
    jm = torch.jit.trace(model, x, check_trace=False)
    out = os.path.join(OUT, f"dswinir_jit_{size}.pt")
    jm.save(out)
    print("JIT ->", out, os.path.getsize(out) // 1024 // 1024, "MB", flush=True)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "weights/w_rain100l_result.pth"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    main(ckpt, size)
