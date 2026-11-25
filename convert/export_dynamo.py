"""ONNX export via torch dynamo exporter (torch>=2.2), lower memory than the
legacy symbolic path that OOMed at 256x256 on this machine.

Usage:
    python3 convert/export_dynamo.py [ckpt] [size]
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
    model.load_state_dict(sd, strict=True)
    del sd
    gc.collect()
    model.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, size, size, dtype=torch.float32)

    out = os.path.join(OUT, f"dswinir_dyn_{size}.onnx")
    print(f"dynamo tracing {size}x{size} ...", flush=True)
    torch.onnx.export(
        model, (x,), out,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
    )
    print("ONNX ->", out, os.path.getsize(out) // 1024 // 1024, "MB", flush=True)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "weights/w_rain100l_result.pth"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    main(ckpt, size)
