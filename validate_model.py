"""Gate G1/G2 for rebuilt MSDeformableNAIR model.

G1: auto-detect config from a checkpoint state_dict, build model, load_state_dict
    strict=True. Prints config + parameter-count report.
G2 (optional, needs data): PSNR spot-check on real test images
    (Rain100L test: 38.19 dB expected for the Rain100L single-task checkpoint).

Usage:
    python3 validate_model.py weights/rain100l.pth
    python3 validate_model.py weights/rain100l_official.pth --input-dir test/input --gt-dir test/target --psnr-target 38.19
"""
import argparse
import glob
import os

import torch
import numpy as np

from model import MSDeformableNAIR


def sd_of(ckpt_path):
    import zipfile
    if zipfile.is_zipfile(ckpt_path):
        with zipfile.ZipFile(ckpt_path) as archive:
            if not any(n.endswith('/data.pkl') or n == 'data.pkl' for n in archive.namelist()):
                raise ValueError(f'{ckpt_path} is a ZIP of results, not a PyTorch checkpoint')
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
        sd = ckpt["params_ema"]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        sd = ckpt["params"]
    else:
        sd = ckpt
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def infer_config(sd):
    """Read architecture sizes straight from state_dict tensor shapes."""
    cfg = {}
    # Conv2d weight: (out, in, kH, kW). patch_embed: in=3 -> out=dim.
    cfg["dim"] = int(sd["patch_embed.proj.weight"].shape[0])
    cfg["inp_channels"] = int(sd["patch_embed.proj.weight"].shape[1])
    cfg["out_channels"] = int(sd["output.weight"].shape[0])

    def stage_len(prefix):
        idxs = {int(k.split(f"{prefix}.")[1].split(".")[0])
                for k in sd if k.startswith(prefix + ".")}
        return max(idxs) + 1 if idxs else 0

    cfg["num_blocks"] = [stage_len("encoder_level1"), stage_len("encoder_level2"),
                         stage_len("encoder_level3"), stage_len("latent")]
    cfg["num_refinement_blocks"] = stage_len("refinement")

    cfg["heads"] = []
    cfg["kernel_size"] = []
    for i, lvl in enumerate(["encoder_level1", "encoder_level2", "encoder_level3", "latent"]):
        p = f"{lvl}.0.attn"
        rpb = f"{p}.rpb_h1"
        cfg["heads"].append(int(sd[rpb].shape[0]) * 3 if rpb in sd else 0)
        ks = [k for k in sd if k.startswith(f"{p}.conv_offset.0.") and k.endswith(".weight")]
        cfg["kernel_size"].append(int(sd[ks[0]].shape[2]))
    # ffn expansion from level-1 ffn.project_in (hidden*2) out shape
    fi = sd["encoder_level1.0.ffn.project_in.weight"].shape[0]  # 2*hidden
    cfg["ffn_expansion_factor"] = int(fi // (2 * cfg["dim"]))
    return cfg


def psnr(a, b):
    a, b = np.asarray(a).astype(np.float64), np.asarray(b).astype(np.float64)
    m = np.mean((a - b) ** 2)
    return float("inf") if m < 1e-12 else 10 * np.log10(1.0 / m)


def load_png(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Cannot read {path}')
    return img[:, :, ::-1].astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--input-dir", help="degraded test images")
    ap.add_argument("--gt-dir", help="separate clean references, same basenames")
    ap.add_argument("--psnr-target", type=float, help="expected mean PSNR to compare against")
    a = ap.parse_args()

    sd = sd_of(a.ckpt)
    cfg = infer_config(sd)
    print("Inferred config:", cfg)
    print("Checkpoint tensors:", len(sd), "| total params:",
          sum(v.numel() for v in sd.values()) / 1e6, "M")

    m = MSDeformableNAIR(**cfg)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("MISSING:", list(missing)[:20])
        print("UNEXPECTED:", list(unexpected)[:20])
        raise SystemExit("G1 FAILED")
    print("G1 PASS: strict key match, 0 missing / 0 unexpected")
    m.eval()

    if bool(a.input_dir) != bool(a.gt_dir):
        ap.error('--input-dir and --gt-dir are required together')
    if a.input_dir:
        if os.path.realpath(a.input_dir) == os.path.realpath(a.gt_dir):
            ap.error('degraded and ground-truth directories must differ')
        pairs = sorted(glob.glob(os.path.join(a.input_dir, "*.png")))
        if not pairs:
            ap.error('no PNG test images found')
        vals = []
        with torch.no_grad():
            for p in pairs:
                img = load_png(p)
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                from tools.restore_tiled import restore_tensor
                out = restore_tensor(m, t).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
                gt = load_png(os.path.join(a.gt_dir, os.path.basename(p)))
                if out.shape != gt.shape:
                    raise ValueError(f'Shape mismatch: {p}')
                vals.append(psnr(out, gt))
        mean = sum(vals) / len(vals)
        print(f"G2: mean PSNR = {mean:.2f} dB over {len(vals)} imgs"
              + (f" (target {a.psnr_target})" if a.psnr_target else ""))
        if a.psnr_target and abs(mean - a.psnr_target) > 0.5:
            print("G2 FAIL: >0.5 dB from target. Check checkpoint/config, preprocessing, "
                  "tiling differences and original CUDA parity before conversion.")
            raise SystemExit(1)
        print("G2 PASS")


if __name__ == "__main__":
    main()
