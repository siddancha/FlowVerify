import numpy as np
import cv2
from easydict import EasyDict as edict
import importlib.util
import yaml
    
def crop_and_resize_target(tg_mask, other_arrays, size, ratio=0.95):
    """Crops target from image given a mask of the target.

    Given a target mask and possibly other arrays, creates a new target mask of size 'size' by cropping the target
    and centering it such that the target occupies 'ratio' fraction of the image. Also applies the same transformation
    to each of the 'other_arrays'.

    Uses cv2.INTER_NEAREST interpolation for resizing. Hence, the resized mask and all other arrays should be exactly
    compatible.

    Args:
        tg_mask (np.ndarray, type=np.uint8, shape=H x W): Binary mask of target in im.
        other_arrays (list(np.ndarray, shape=H x W x C_i)): List of other arrays. Each array can have a different number
            of channels (RGB images will have 3, flow will have 2), but spatial dimensions have to be the same.
        size: size of final target image.
        ratio: ratio of target in final image.
    Returns:
        mask_f (np.ndarray, type=np.uint8): binary mask of final image.
        other_arrays (list(np.ndarray, shape=size x size x C_i)): List of cropped and resized other arrays.
    """
    for array in other_arrays:
        assert(tg_mask.shape[0] == array.shape[0] and tg_mask.shape[1] == array.shape[1])

    # Calculate extents of target in initial image [min, max).
    x_inds, y_inds = tg_mask.max(axis=0).nonzero()[0], tg_mask.max(axis=1).nonzero()[0]
    xmin_i, ymin_i = x_inds.min(), y_inds.min()
    xmax_i, ymax_i = x_inds.max() + 1, y_inds.max() + 1
    # Calculate extents of target in final image [min, max).
    tg_w, tg_h = xmax_i - xmin_i, ymax_i - ymin_i
    scale = min((ratio * size) / tg_w, (ratio * size) / tg_h)
    tg_w, tg_h = int(scale * tg_w), int(scale * tg_h)

    xmin_f, ymin_f = (size - tg_w) // 2, (size - tg_h) // 2
    xmax_f, ymax_f = xmin_f + tg_w, ymin_f + tg_h

    # Resize and paste mask.
    tg_mask_f = np.zeros((size, size), dtype=np.uint8)
    tg_mask_cropped = tg_mask[ymin_i:ymax_i, xmin_i:xmax_i]
    tg_mask_resized = cv2.resize(tg_mask_cropped, (tg_w, tg_h), interpolation=cv2.INTER_NEAREST)
    tg_mask_f[ymin_f:ymax_f, xmin_f:xmax_f] = tg_mask_resized

    # Resize and paste other images.
    for i, array in enumerate(other_arrays):
        array = array * tg_mask[:, :, None].repeat(array.shape[2], axis=2)  # background fill is 0
        array_f = np.zeros((size, size, array.shape[2]), dtype=array.dtype)  # background fill is 0
        array_cropped = array[ymin_i:ymax_i, xmin_i:xmax_i, :]
        array_resized = cv2.resize(array_cropped, (tg_w, tg_h), interpolation=cv2.INTER_NEAREST)
        array_f[ymin_f:ymax_f, xmin_f:xmax_f, :] = array_resized
        other_arrays[i] = array_f

    return tg_mask_f, other_arrays


def load_config(cfg_path):
    if cfg_path[-5:] == '.yaml':
        print('Loading .yaml file ... ', end='', flush=True)
        with open(cfg_path, 'r') as f:
            cfg = edict(yaml.load(f))
        print('done.')
        return cfg
    # TODO: Remove .py config loader after migration to .yaml complete.
    spec = importlib.util.spec_from_file_location('cfg', cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg
