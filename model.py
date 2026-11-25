"""Self-contained MSDeformableNAIR model (DSwinIR).

Standalone port of basicsr/archs/MSDeformableNAIR.py + deformable_self_attention.py:
- no `basicsr`, `natten`, `timm`, `einops` deps
- NATTEN na2d_qk/na2d_av replaced by pure-torch shifted boundary windows
  (see natten_pure.py; full CUDA parity still requires verification).
LayerNorm2d follows BasicSR arch_util (channel dim, 4D (1,C,1,1) affine).
"""
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from natten_pure import na2d_qk, na2d_av


class LayerNorm2d(nn.Module):
    """BasicSR/NAFNet LayerNorm2d: normalize channels at each pixel.

    1D affine (weight/bias shape (C,)) — matches the repo checkpoint keys
    (e.g. `encoder_level1.0.norm1.body.weight` has shape (48,)).
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        return x * w + b


def to_3d(x):
    return x.permute(0, 2, 3, 1).view(x.shape[0], -1, x.shape[1])


def to_4d(x, h, w):
    return x.view(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.norm_type = LayerNorm_type
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = LayerNorm2d(dim, eps=1e-5)

    def forward(self, x):
        if self.norm_type == 'BiasFree':
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        return self.body(x)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MSDeformableNeighborhoodAttention(nn.Module):
    """Multi-scale deformable neighborhood attention (DSwinIR).

    Identical structure to the repo class, but the NATTEN calls
    (na2d_qk/na2d_av) use the pure-torch local-window implementation.
    Channel-last (B, G, H, W, C) layout as in the original.
    """

    def __init__(self, dim, num_heads, kernel_size, dilation=1,
                 offset_range_factor=1.0, stride=1, use_pe=True, dwc_pe=True,
                 no_off=False, fixed_pe=False, is_causal=False,
                 rel_pos_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        n_head_channels = dim // num_heads
        n_groups = num_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = num_heads
        self.nc = n_head_channels * num_heads
        self.n_groups = num_heads
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = kernel_size
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels,
                      kk, stride, pad_size, groups=self.n_group_channels),
            LayerNorm2d(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, 1, 1, 0)

        if rel_pos_bias:
            self.rpb_h1 = nn.Parameter(torch.zeros(self.n_heads // 3, 2 * kernel_size - 1, 2 * kernel_size - 1))
            nn.init.trunc_normal_(self.rpb_h1, std=0.02, a=-2.0, b=2.0)
            self.rpb_h2 = nn.Parameter(torch.zeros(self.n_heads // 3, 2 * kernel_size - 1, 2 * kernel_size - 1))
            nn.init.trunc_normal_(self.rpb_h2, std=0.02, a=-2.0, b=2.0)
            self.rpb_h3 = nn.Parameter(torch.zeros(self.n_heads // 3, 2 * (kernel_size + 2) - 1, 2 * (kernel_size + 2) - 1))
            nn.init.trunc_normal_(self.rpb_h3, std=0.02, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb_h1", None)
            self.register_parameter("rpb_h2", None)
            self.register_parameter("rpb_h3", None)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.rpe_table = nn.Conv2d(self.nc, self.nc, 3, 1, 1, groups=self.nc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij')
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = q.reshape(B, self.n_groups, self.n_group_channels, H, W).permute(0, 1, 3, 4, 2)
        q_off = q_off.permute(0, 1, 4, 2, 3).reshape(B * self.n_groups, self.n_group_channels, H, W)
        offset = self.conv_offset(q_off).contiguous()

        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = offset.permute(0, 2, 3, 1)  # b p h w -> b h w p
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)

        x_sampled = x_sampled.reshape(B, C, H, W)

        residual_lepe = self.rpe_table(q)

        k = self.proj_k(x_sampled)
        v = self.proj_v(x_sampled)

        def to_cl(t):
            return t.reshape(B, self.n_groups, self.n_group_channels, H, W).permute(0, 1, 3, 4, 2)

        q_cl = to_cl(q) * self.scale
        k_cl = to_cl(k)
        v_cl = to_cl(v)

        q1, q2, q3 = q_cl.chunk(chunks=3, dim=1)
        k1, k2, k3 = k_cl.chunk(chunks=3, dim=1)
        v1, v2, v3 = v_cl.chunk(chunks=3, dim=1)

        def run(attn_in_q, attn_in_k, attn_in_v, kernel_size, dilation, rpb):
            s = na2d_qk(attn_in_q, attn_in_k, kernel_size=kernel_size,
                        dilation=dilation, is_causal=self.is_causal, rpb=rpb)
            a = s.softmax(dim=-1)
            a = self.attn_drop(a)
            return na2d_av(a, attn_in_v, kernel_size=kernel_size,
                           dilation=dilation, is_causal=self.is_causal)

        out_h1 = run(q1, k1, v1, self.kernel_size, self.dilation, self.rpb_h1)
        out_h2 = run(q2, k2, v2, self.kernel_size, 2, self.rpb_h2)
        out_h3 = run(q3, k3, v3, (self.kernel_size[0] + 2, self.kernel_size[1] + 2), self.dilation, self.rpb_h3)

        out = torch.cat([out_h1, out_h2, out_h3], dim=1)  # B g H W C
        out = out.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        y = self.proj_drop(self.proj_out(out))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor,
                 bias, LayerNorm_type, rel_pos_bias, kernel_size=3, dilation=1):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MSDeformableNeighborhoodAttention(
            dim, num_heads, kernel_size=kernel_size, dilation=dilation, rel_pos_bias=rel_pos_bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class MSDeformableNAIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[4, 6, 6, 8],
                 kernel_size=[3, 3, 3, 3],
                 dilation=[1, 1, 1, 1],
                 num_refinement_blocks=4,
                 heads=[2, 4, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 rel_pos_bias=True,
                 dual_pixel_task=False,
                 global_residual=True,
                 ):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, rel_pos_bias=rel_pos_bias,
                             kernel_size=kernel_size[0], dilation=dilation[0])
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[1], rel_pos_bias=rel_pos_bias,
                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                            kernel_size=kernel_size[1], dilation=dilation[1])
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**2), num_heads=heads[2], rel_pos_bias=rel_pos_bias,
                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                            kernel_size=kernel_size[2], dilation=dilation[2])
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**3), num_heads=heads[3], rel_pos_bias=rel_pos_bias,
                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                            kernel_size=kernel_size[3], dilation=dilation[3])
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**2), num_heads=heads[2], rel_pos_bias=rel_pos_bias,
                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                            kernel_size=kernel_size[2], dilation=dilation[2])
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[1], rel_pos_bias=rel_pos_bias,
                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                            kernel_size=kernel_size[1], dilation=dilation[1])
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[0], rel_pos_bias=rel_pos_bias,
                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                            kernel_size=kernel_size[0], dilation=dilation[0])
            for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[0],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias, LayerNorm_type=LayerNorm_type, rel_pos_bias=rel_pos_bias,
                            kernel_size=kernel_size[0], dilation=dilation[0])
            for i in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), 1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, 3, 1, 1, bias=bias)
        self.global_residual = global_residual

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        if self.global_residual:
            out_dec_level1 = out_dec_level1 + inp_img

        return out_dec_level1


def build_dswinir(cfg=None, checkpoint_path=None, **overrides):
    """Build model, (optionally) set dims from checkpoint shapes, load weights strict.

    cfg keys = MSDeformableNAIR __init__ args. `checkpoint_path` (if given) loads
    state_dict strict=True and raises with a key-diff report on mismatch.
    """
    import json, os
    kwargs = {
        "inp_channels": 3, "out_channels": 3, "dim": 32,
        "num_blocks": [4, 6, 6, 8], "kernel_size": [3, 3, 3, 3],
        "dilation": [1, 1, 1, 1], "num_refinement_blocks": 4,
        "heads": [2, 4, 8, 8], "ffn_expansion_factor": 2, "bias": False,
        "LayerNorm_type": "WithBias", "rel_pos_bias": True,
        "dual_pixel_task": False, "global_residual": True,
    }
    if cfg:
        kwargs.update(cfg)
    kwargs.update(overrides)
    model = MSDeformableNAIR(**kwargs)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "params_ema" in ckpt:
            sd = ckpt["params_ema"]
        elif isinstance(ckpt, dict) and "params" in ckpt:
            sd = ckpt["params"]
        else:
            sd = ckpt
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad_shape = [k for k in sd if k in model.state_dict() and tuple(sd[k].shape) != tuple(model.state_dict()[k].shape)]
        if missing or unexpected or bad_shape:
            print(f"MISSING ({len(missing)}): {list(missing)[:10]}")
            print(f"UNEXPECTED ({len(unexpected)}): {list(unexpected)[:10]}")
            print(f"BAD SHAPE: {bad_shape[:10]}")
            raise SystemExit(1)
    model.eval()
    return model


if __name__ == "__main__":
    # smoke test: forward 128x128. heads must be %3==0 (rpb shaped n_heads//3);
    # actual checkpoint heads read from state_dict rpb_h1 shape.
    m = MSDeformableNAIR(dim=24, num_blocks=[2, 2, 2, 2], num_refinement_blocks=2,
                         heads=[3, 6, 12, 12])
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        y = m(x)
    print("out", tuple(y.shape), y.dtype, float(y.abs().max()))
    print("params", sum(p.numel() for p in m.parameters()) / 1e6, "M")
