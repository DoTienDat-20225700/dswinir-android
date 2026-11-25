"""Trace-only ONNX export + hand-rolled .npy ref dump.

Runs under the SYSTEM python (torch 2.2.2, where numpy is ABI-broken in-process).
So refs are written as .npy via a tiny manual writer (no numpy import), giving the
venv (which has numpy but no torch) files it can load directly for the ORT check.

Writes into convert/out/:
    dswinir_<size>.onnx
    dswinir_<size>_input.npy   (NCHW float32, the traced input)
    dswinir_<size>_ref.npy     (NCHW float32, the torch reference output)

Usage:
    python3 convert/_export_bridge.py [ckpt] [size]
"""
import gc
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_model import sd_of, infer_config
from model import MSDeformableNAIR

OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"


def save_npy(path, arr):
    """Minimal .npy writer (C-order float32), no numpy/torch->numpy dependency.

    Layout: b'\\x93NUMPY' + b'\\x01\\x00' version + <H> header_len + header + data.
    `header_len` is the byte length of the header field (dict + space-padding +
    trailing newline); 10 + header_len must be a multiple of 64. Data bytes come
    from struct.pack of the float values (Tensor.numpy() is ABI-broken here).
    """
    a = torch.as_tensor(arr, dtype=torch.float32).contiguous()
    shape = tuple(a.shape)
    flat = a.reshape(-1).tolist()
    data = struct.pack(f"<{len(flat)}f", *flat)
    shape_str = ",".join(str(s) for s in shape)
    pydict = "{'descr': '<f4', 'fortran_order': False, 'shape': (%s,), }" % shape_str
    L = len(pydict)
    pad = (-(11 + L)) % 64
    field = pydict.encode("ascii") + b" " * pad + b"\n"
    assert (10 + len(field)) % 64 == 0, (10 + len(field))
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")
        f.write(b"\x01\x00")  # version major=1, minor=0 (2 bytes)
        f.write(struct.pack("<H", len(field)))
        f.write(field)
        f.write(data)
    print(f"wrote {os.path.basename(path)} shape={shape} bytes={len(data)}", flush=True)


def main(ckpt_path, size):
    os.makedirs(OUT, exist_ok=True)
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

    # reference output FIRST, so we still have a ref even if the trace OOMs
    with torch.no_grad():
        ref = model(x)
    assert torch.isfinite(ref).all(), "non-finite ref output"
    print("ref out range:", float(ref.min()), float(ref.max()), flush=True)
    save_npy(os.path.join(OUT, f"dswinir_{size}_input.npy"), x)
    save_npy(os.path.join(OUT, f"dswinir_{size}_ref.npy"), ref)
    del ref  # keep x - the trace reuses it
    gc.collect()

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
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    main(ckpt, size)
