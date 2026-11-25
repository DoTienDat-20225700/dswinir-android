import unittest
import torch
from model import LayerNorm2d, MSDeformableNeighborhoodAttention
from natten_pure import na2d_qk, na2d_av


class SemanticsTest(unittest.TestCase):
    def test_layernorm_channels(self):
        x = torch.randn(2, 4, 5, 7)
        m = LayerNorm2d(4)
        expected = torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1), (4,), eps=1e-5)
        torch.testing.assert_close(m(x).permute(0, 2, 3, 1), expected)

    def test_boundary_and_dilation_against_scalar_reference(self):
        # Independent reference: enumerate the residue sub-lattice, then select
        # the centered window, shifting it fully inside the image at boundaries.
        torch.manual_seed(1)
        for h, w, d in [(5, 7, 1), (7, 9, 2), (9, 7, 2)]:
            q, k, v = [torch.randn(1, 2, h, w, 3) for _ in range(3)]
            bias = torch.randn(2, 5, 5)
            scores = na2d_qk(q, k, 3, d, rpb=bias)
            ref = torch.empty_like(scores)
            val = torch.empty_like(v)
            for y in range(h):
                ys = list(range(y % d, h, d)); yp = ys.index(y)
                yy = ys[max(0, min(yp-1, len(ys)-3)):][:3]
                for x in range(w):
                    xs = list(range(x % d, w, d)); xp = xs.index(x)
                    xx = xs[max(0, min(xp-1, len(xs)-3)):][:3]
                    for a, ky in enumerate(yy):
                        for b, kx in enumerate(xx):
                            ref[:, :, y, x, a*3+b] = (q[:, :, y, x]*k[:, :, ky, kx]).sum(-1) + bias[:, 2+(ky-y)//d, 2+(kx-x)//d]
                    probs = ref[:, :, y, x].softmax(-1)
                    val[:, :, y, x] = sum(probs[..., a*3+b, None]*v[:, :, ky, kx] for a, ky in enumerate(yy) for b, kx in enumerate(xx))
            torch.testing.assert_close(scores, ref)
            torch.testing.assert_close(na2d_av(scores.softmax(-1), v, 3, d), val)

    def test_offset_coordinates_preserve_channels(self):
        # Capture sampling grid to ensure dy/dx are transposed, never reshaped.
        from unittest.mock import patch
        m = MSDeformableNeighborhoodAttention(6, 3, 3)
        x = torch.randn(1, 6, 8, 8)
        q = m.proj_q(x).reshape(3, 2, 8, 8)
        offset = m.conv_offset(q).tanh() / 7
        expected = offset.permute(0, 2, 3, 1) + m._get_ref_points(8, 8, 1, x.dtype, x.device)
        original = torch.nn.functional.grid_sample
        grids = []
        def sample(*args, **kw):
            grids.append(kw['grid'].detach())
            return original(*args, **kw)
        with patch('model.F.grid_sample', side_effect=sample):
            m(x)
        torch.testing.assert_close(grids[0], expected[..., (1, 0)])


if __name__ == '__main__':
    unittest.main()
