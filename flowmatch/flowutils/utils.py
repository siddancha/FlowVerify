import numpy as np


def get_identity_flow(width, height):
    """Create an identity flow grid with flow valus in [0, 1]
    Args:
        width (int): Width of grid.
        height (int): Height of grid.
    Returns:
        grid (np.ndarray, shape=(height, width, 2), dtype=np.float64): Identity flow field with (u, v) values in last
            dimension (0 <= u,v <= 1).
    Examples:
        >>> flow = get_identity_flow(3, 2)
        >>> flow[:, :, 0]
        array([[0.16666667, 0.5       , 0.83333333],
               [0.16666667, 0.5       , 0.83333333]])
        >>> flow[:, :, 1]
        array([[0.25, 0.25, 0.25],
               [0.75, 0.75, 0.75]])
    """
    # Grid contains coordinates of centers of pixels in tg_bbox, in the range [0, 1].
    grid = np.indices((height, width)).transpose((1, 2, 0)) + 0.5
    grid = np.flip(grid, 2)  # flip so that grid[:, :, 0] is u and grid[:, :, 1] is v
    grid /= [width, height]  # bring grid values in the range [0, 1]
    return grid
