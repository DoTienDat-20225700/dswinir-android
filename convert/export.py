"""Export the rebuilt MSDeformableNAIR model to ONNX (opset 17, fp32, 256x256)
and dump a torch reference output for later TFLite cross-check.

Uses torch.save (no numpy — the installed torch 2.2.2 is ABI-incompatible with
system numpy 2.x on this machine, so any numpy call in-process raises "Numpy is
not available").

Usage:
    python3 convert/export.py weights/w_rain100l_result.pth
Outputs:
    convert/out/dswinir.onnx
    convert/out/ref_input.pt  ref_output.pt  (fp32, NCHW; ref input is a fixed
    deterministic rand(0,1) image, not a real photo)
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_model import sd_of, infer_config
from model import MSDeformableNAIR


def main(ckpt_path):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)

    sd = sd_of(ckpt_path)
    cfg = infer_config(sd)
    print("config:", cfg)
    model = MSDeformableNAIR(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (list(missing)[:5], list(unexpected)[:5])
    model.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        ref = model(x)
    assert torch.isfinite(ref).all(), "non-finite output"
    print("ref out range:", float(ref.min()), float(ref.max()))
    torch.save(x, os.path.join(out_dir, "ref_input.pt"))
    torch.save(ref, os.path.join(out_dir, "ref_output.pt"))

    # fixed 256x256 trace: the DeformableAttn offset_range constant is baked at
    # trace-time, so dynamic H/W would be numerically wrong. The app tiles input
    # to 256x256 and pads; no dynamic axes needed.
    onnx_path = os.path.join(out_dir, "dswinir.onnx")
    torch.onnx.export(
        model, (x,), onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=False,
    )
    print("ONNX ->", onnx_path, os.path.getsize(onnx_path) // 1024 // 1024, "MB")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        o = sess.run(None, {"input": x.numpy()})[0]
        import numpy as _np
        d = float(_np.abs(o - ref.numpy()).max())
        print(f"ONNX vs torch max abs diff = {d:.3e}")
        assert d < 1e-3, "ONNX/torch divergence too large"
        print("ONNX GATE PASS")
    except ImportError:
        print("onnxruntime not installed — skip cross-check")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights/w_rain100l_result.pth")
