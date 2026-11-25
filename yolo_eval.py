"""YOLO detection quality before vs. after DSwinIR restoration.

Pipeline:
    clean.png --(synthetic rain degradation)--> degraded.png
    degraded.png --DSwinIR--> restored.png
    YOLO(degraded) vs YOLO(restored)  ->  detection report

Synthetic degradation keeps the test self-contained (no rain dataset needed).
The torch model runs directly here (no tiling); tiling is a concern for the
ONNX/TFLite app export only.

Usage:
    python3 yolo_eval.py --clean path/to/image.png [--out outdir]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from model import MSDeformableNAIR
from validate_model import sd_of, infer_config

YCONF = 0.25
IOU_MATCH = 0.5


def add_rain(img, intensity=0.5, streaks=150, seed=0):
    """Cheap synthetic rain: darken + gaussian blur + bright streaks."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * (1.0 - 0.4 * intensity)
    out = cv2.GaussianBlur(out, (5, 5), 0)
    for _ in range(streaks):
        x = rng.integers(0, out.shape[1])
        y = rng.integers(0, out.shape[0])
        L = rng.integers(8, 25)
        cv2.line(out, (x, y), (x + 2, y + L), (210, 220, 230), thickness=1, lineType=cv2.LINE_AA)
    return out.clip(0, 255).astype(np.uint8)


def yolo_dets(img_bgr, model, conf=YCONF):
    """Run ultralytics YOLO on a BGR image -> list of [x1,y1,x2,y2,conf,cls]."""
    r = model(img_bgr, conf=conf, verbose=False)[0]
    boxes, confs, cls = r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()
    out = []
    for i in range(len(boxes)):
        b = boxes[i].tolist()
        out.append([b[0], b[1], b[2], b[3], float(confs[i]), int(cls[i])])
    return out


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - iw*ih
    return iw*ih/u if u > 0 else 0.0


def greedy_match(before, after, thr=IOU_MATCH):
    """Pair each before-box with the best-matching after-box (IoU>thr).
    Returns (n_before, n_after, n_matched, iou_list)."""
    used = [False]*len(after)
    pairs = []
    for b in before:
        best, best_iou = -1, 0.0
        for j, a in enumerate(after):
            if used[j] or a[5] != b[5]:
                continue
            v = iou(b, a)
            if v > best_iou:
                best, best_iou = j, v
        if best >= 0 and best_iou > thr:
            used[best] = True
            pairs.append(best_iou)
    return len(before), len(after), len(pairs), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--ckpt", default="weights/w_rain100l_result.pth")
    ap.add_argument("--out", default="yolo_out")
    ap.add_argument("--rain-int", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # load DSwinIR torch model
    sd = sd_of(a.ckpt)
    m = MSDeformableNAIR(**infer_config(sd))
    m.load_state_dict(sd, strict=True)
    m.eval()

    # load YOLO (auto-download ~5MB yolo11n)
    from ultralytics import YOLO
    ymodel = YOLO("yolo11n.pt")

    clean_bgr = cv2.imread(a.clean)
    assert clean_bgr is not None, f"cannot read {a.clean}"
    rain = add_rain(clean_bgr, intensity=a.rain_int)

    # DSwinIR runs on [0,1] NCHW
    def restore(img_bgr):
        t = torch.from_numpy(img_bgr[:, :, ::-1].astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            o = m(t).squeeze(0)
        o = o.permute(1, 2, 0).cpu().numpy() * 255.0
        return np.clip(o, 0, 255).astype(np.uint8)

    restored = restore(rain)

    cv2.imwrite(os.path.join(a.out, "degraded.png"), rain)
    cv2.imwrite(os.path.join(a.out, "restored.png"), restored)

    d_clean = yolo_dets(clean_bgr, ymodel)
    d_degr = yolo_dets(rain, ymodel)
    d_rest = yolo_dets(restored, ymodel)

    nb, na, nm, ious = greedy_match(d_degr, d_rest)
    nc_clean = len(d_clean)
    report = {
        "clean_dets": nc_clean,
        "degraded_dets": nb,
        "restored_dets": na,
        "matched": nm,
        "recall_vs_degraded": nm/nb if nb else None,
        "mean_iou_matched": sum(ious)/len(ious) if ious else None,
        "mean_conf": {
            "clean": np.mean([d[4] for d in d_clean]) if d_clean else 0.0,
            "degraded": np.mean([d[4] for d in d_degr]) if d_degr else 0.0,
            "restored": np.mean([d[4] for d in d_rest]) if d_rest else 0.0,
        },
    }
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # annotated side-by-side
    def draw(img, dets, color, label):
        im = img.copy()
        for d in dets:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im, f"{label} {d[4]:.2f}", (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return im
    side = np.hstack([draw(clean_bgr, d_clean, (0,255,0), "clean"),
                      draw(rain, d_degr, (0,0,255), "degraded"),
                      draw(restored, d_rest, (255,160,0), "restored")])
    cv2.imwrite(os.path.join(a.out, "side_by_side.png"), side)
    print("wrote", a.out, "/side_by_side.png")


if __name__ == "__main__":
    main()
