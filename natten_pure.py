"""Unfused noncausal NATTEN 0.17 semantics, in exportable PyTorch.

Boundary windows shift inward on each dilation residue lattice. They are NOT
zero padded. Formula follows NATTEN v0.17.1 natten_commons.cuh get_window_start
and get_pb_start. Validate against the original CUDA kernels before publishing
converted models. CHUNK changes peak working memory, not numerical semantics.
"""
import os
import torch

CHUNK = int(os.environ.get('NA2D_CHUNK', '4'))
if CHUNK < 1:
    raise ValueError('NA2D_CHUNK must be positive')


def _axis(length, kernel, dilation, device):
    if length < kernel * dilation:
        raise ValueError(f'NATTEN needs length >= kernel*dilation: {length} < {kernel*dilation}')
    i = torch.arange(length, device=device)
    residue = i % dilation
    count = (length - 1 - residue) // dilation + 1
    # Clip window start in the sub-sequence i % dilation, preserving residue.
    start = torch.minimum(torch.clamp(i // dilation - kernel // 2, min=0), count - kernel)
    start = start * dilation + residue
    pb = kernel - 1 - (i - start) // dilation
    return start, pb


def _indices(h, w, kernel_size, dilation, device):
    kr, kc = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sy, py = _axis(h, kr, dilation, device)
    sx, px = _axis(w, kc, dilation, device)
    taps = [(a, b) for a in range(kr) for b in range(kc)]
    return sy, sx, py, px, taps


def _gather(t, sy, sx, taps, dilation):
    b, g, h, w, c = t.shape
    ids = torch.stack([(sy[:, None] + a*dilation)*w + sx[None, :] + z*dilation
                       for a, z in taps], dim=-1)
    return t.reshape(b, g, h*w, c).index_select(2, ids.reshape(-1)).reshape(b, g, h, w, len(taps), c)


def na2d_qk(q, k, kernel_size, dilation=1, is_causal=False, rpb=None):
    if is_causal:
        raise ValueError('Only noncausal attention is supported')
    _, _, h, w, _ = q.shape
    sy, sx, py, px, taps = _indices(h, w, kernel_size, dilation, q.device)
    chunks = []
    for i in range(0, len(taps), CHUNK):
        sel = taps[i:i+CHUNK]
        scores = (q.unsqueeze(-2) * _gather(k, sy, sx, sel, dilation)).sum(-1)
        if rpb is not None:
            bias = torch.stack([rpb[:, py[:, None]+a, px[None, :]+b] for a, b in sel], dim=-1)
            scores = scores + bias.unsqueeze(0)
        chunks.append(scores)
    return torch.cat(chunks, dim=-1)


def na2d_av(attn, v, kernel_size, dilation=1, is_causal=False):
    if is_causal:
        raise ValueError('Only noncausal attention is supported')
    _, _, h, w, _ = v.shape
    sy, sx, _, _, taps = _indices(h, w, kernel_size, dilation, v.device)
    out = torch.zeros_like(v)
    for i in range(0, len(taps), CHUNK):
        sel = taps[i:i+CHUNK]
        out = out + (attn[..., i:i+len(sel), None] * _gather(v, sy, sx, sel, dilation)).sum(-2)
    return out
