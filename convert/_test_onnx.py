import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from validate_model import sd_of, infer_config
from model import MSDeformableNAIR

size = int(sys.argv[1]) if len(sys.argv) > 1 else 128
sd = sd_of('weights/w_rain100l_result.pth')
cfg = infer_config(sd)
m = MSDeformableNAIR(**cfg); m.load_state_dict(sd, strict=False); m.eval()
x = torch.rand(1, 3, size, size)
with torch.no_grad(): ref = m(x)
out = f'/tmp/dswinir_{size}.onnx'
torch.onnx.export(m, (x,), out, opset_version=17, input_names=['input'],
                  output_names=['output'], do_constant_folding=False)
print(f'ONNX {size} OK {os.path.getsize(out)//1024//1024} MB')
