"""Trace-only ONNX export (no ref recompute — ref_input.pt/ref_output.pt exist).

Split out of export.py because on a low-free-RAM machine the ref forward plus
the ONNX trace together OOMs. Frees the state_dict and GCs before tracing;
CHUNK in natten_pure (4) caps activation memory.

Usage:
    python3 convert/export_onnx_only.py [ckpt] [size]
    size defaults to 256. The DeformableAttn offset_range constant is baked
    at trace-time, so each trace is valid only for its spatial size.
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

    torch.manual_seed(0)
    x = torch.rand(1, 3, size, size, dtype=torch.float32)

    onnx_path = os.path.join(OUT, f"dswinir_{size}.onnx")
    print(f"tracing {size}x{size} ...", flush=True)
    torch.onnx.export(
        model, (x,), onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=False,
    )
    print("ONNX ->", onnx_path, os.path.getsize(onnx_path) // 1024 // 1024, "MB", flush=True)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "weights/w_rain100l_result.pth"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    main(ckpt, size)
