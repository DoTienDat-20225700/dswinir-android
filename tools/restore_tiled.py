"""Same 256-pixel grid and replicate padding used by the Android app."""
import torch
import torch.nn.functional as F


def restore_tensor(model, image, tile=256):
    if image.ndim != 4 or image.shape[0:2] != (1, 3):
        raise ValueError('Expected [1,3,H,W] RGB in [0,1]')
    height, width = image.shape[-2:]
    result = torch.empty_like(image)
    with torch.inference_mode():
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                patch = image[:, :, y:y+tile, x:x+tile]
                h, w = patch.shape[-2:]
                padded = F.pad(patch, (0,tile-w,0,tile-h), mode='replicate')
                output = model(padded)
                if output.shape != padded.shape or not torch.isfinite(output).all():
                    raise ValueError('Invalid restoration output')
                result[:, :, y:y+h, x:x+w] = output[:, :, :h, :w]
    return result
