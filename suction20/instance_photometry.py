"""Exact source-instance color augmentation restricted to the visible bounding box."""
import numpy as np
from PIL import Image, ImageEnhance


def apply_source_rgb(observation, parameters, *, prepare_support=True):
    factors, order, hue, grayscale = parameters
    raw = np.asarray(observation['rgb'], dtype=np.float32)
    mask = np.asarray(observation['mask'], dtype=bool)
    if 'rgb_valid' in observation:
        valid = mask & np.asarray(observation['rgb_valid'], dtype=bool)
    else:
        valid = mask & np.isfinite(raw).all(-1)
    valid = np.asarray(valid, dtype=bool)
    result = np.zeros_like(raw)
    ys = np.flatnonzero(valid.any(axis=1))
    if len(ys):
        xs = np.flatnonzero(valid.any(axis=0))
        box = np.s_[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
        selected = valid[box]
        pixels = np.nan_to_num(raw[box], nan=0., posinf=0., neginf=0.)
        pixels[~selected] = 0.
        image = Image.fromarray(np.round(np.clip(pixels, 0., 1.) * 255).astype(np.uint8))
        for idx in order:
            if idx == 1:
                # Brightness and saturation preserve black up to the contrast
                # transform. The discarded area therefore adds zero to the
                # full-camera luminance sum used by PIL.ImageEnhance.Contrast.
                luminance = np.asarray(image.convert('L')).sum(dtype=np.uint64)
                mean = int(float(luminance) / valid.size + .5)
                gray = Image.new('RGB', image.size, (mean, mean, mean))
                image = Image.blend(gray, image, factors[1])
            elif idx == 0:
                image = ImageEnhance.Brightness(image).enhance(factors[0])
            else:
                image = ImageEnhance.Color(image).enhance(factors[2])
        hsv = np.asarray(image.convert('HSV')).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue) % 256
        image = Image.fromarray(hsv, mode='HSV').convert('RGB')
        if grayscale:
            image = image.convert('L').convert('RGB')
        crop = np.asarray(image, dtype=np.float32) / 255.
        crop[~selected] = 0.
        result[box] = crop
    if not prepare_support:
        # Route B consumes image RGB. Do not materialize an unused second copy
        # of those pixels as flattened support colors.
        return dict(observation, rgb=result, rgb_valid=valid)
    uv = np.asarray(observation['support_uv'], np.int64)
    return dict(observation, rgb=result, rgb_valid=valid,
                support_rgb=result[uv[:, 1], uv[:, 0]].copy(),
                support_rgb_valid=valid[uv[:, 1], uv[:, 0]].copy())
