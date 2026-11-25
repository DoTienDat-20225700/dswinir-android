"""ONNX export via the torch FX/dynamo exporter (torch 2.2.2 API).

Different path from the legacy `torch.onnx.export` symbolic pipeline that
OOMed/swap-thrashed at 128x128 on this machine:
  - no JIT symbolic trace, no legacy C++ shape-inference pass over the
    unrolled tap graph
  - fx graph + onnxscript per-op ONNX generation (opset 18 is fixed in
    torch 2.2.2; do NOT pass opset_version as a kwarg - it leaks into
    model.forward())
Requires onnxscript (pip install onnxscript in the same interpreter).

Usage:
    python3 convert/_dyn_export.py [ckpt] [size]
Writes convert/out/dswinir_dyn_<size>.onnx
"""
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_model import sd_of, infer_config
from model import MSDeformableNAIR

OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"


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

    from torch.onnx import ExportOptions
    print(f"dynamo_export {size}x{size} (opset 18, static shapes) ...", flush=True)
    prog = torch.onnx.dynamo_export(
        model, (x,),
        export_options=ExportOptions(dynamic_shapes=False),
    )
    out = os.path.join(OUT, f"dswinir_dyn_{size}.onnx")
    prog.save(out)
    print("ONNX ->", out, os.path.getsize(out) // 1024 // 1024, "MB", flush=True)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "weights/w_rain100l_result.pth"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    main(ckpt, size)
