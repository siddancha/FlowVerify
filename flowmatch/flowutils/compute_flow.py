import numpy as np
from flowmatch.flowutils.utils import get_identity_flow

def affine_flow(tg_size, tg_bbox, rmat, GLOBAL=False):
    """Creates ground truth flow for an affine transformation (rotate + translate) of target.

    Creates a flow field of tg_bbox in tg_image undergoing an affine transformation defined by rmat.
    Flow for all pixels in tg_image that are outside tg_bbox is 0.

    Args:
        tg_size (tuple[int], size=2): Size of target image (width, height).
        tg_bbox (tuple[int], size=4): Bbox of target in tg_image, in xyxy format, endpoints inclusive.
        rmat (np.ndarray, np.float32, size=(3, 3)): Rotation matrix of transformation that maps (u, v) of tg_bbox.

    Returns:
        flow (np.ndarray, type=float32, size=HxWx2): A flow array with same width and height of tg_image, and two
            channels for u and v respectively. Values are in the range [0, 1] denoting coordinates relative to tg_image.
    """
    # assert(np.all(rmat[2] == [0., 0., 1.]))
    tg_x1, tg_y1, tg_x2, tg_y2 = tg_bbox
    tg_w, tg_h = tg_x2 - tg_x1 + 1, tg_y2 - tg_y1 + 1

    # Grid contains coordinates of centers of pixels in tg_bbox, in the range [0, 1].
    grid = get_identity_flow(tg_w, tg_h)
    if not GLOBAL:
        grid *= [tg_w, tg_h]

    # Append ones to (u, v) so that it can be multiplied by rotation matrix.
    grid_input = np.concatenate([grid, np.ones((tg_h, tg_w, 1), dtype=np.float32)], axis=2)

    # Multiply rotation matrix with grid_input to get affine transformation.
    # np.dot(a, b) does sum-product over last axis of a and second-to-last axis of b.
    # rmat is to be pre-multiplied with a column vector. This means the sum-product of the ith row with input vector
    # gives ith coordinate of the transformed vector. The rows of rmat need to form the second-to-last axis. Hence, we
    # take transpose of rmat.
    flow_tb = np.dot(grid_input, rmat.T)  # acronym means flow for tg_bbox
    # Remove the extra 1s.
    # assert(np.all(flow_tb[:, :, 2] == 1.))
    flow_tb = np.divide(flow_tb[:, :, :2], np.dstack([flow_tb[:,:,2], flow_tb[:,:,2]]))
    if not GLOBAL:
        flow_tb /= [tg_w, tg_h]

    # Place grid as part of a larger flow array.
    flow = np.zeros((tg_size[1], tg_size[0], 2), dtype=np.float32)
    flow[tg_y1:tg_y2+1, tg_x1:tg_x2+1, :] = flow_tb
    return flow


def linear_flow(tg_size, tg_bbox, cs_bbox):
    """Creates ground truth flow for an axis-parallel linear transform of target.

    Creates a flow field of tg_bbox in tg_image going to cs_bbox in a cropped scene image. Assumption is that the
    transformation is linear and axis parallel. Flow for all pixels in tg_image that are outside tg_bbox is 0.

    Args:
        tg_size (tuple[int], size=2): Size of target image (width, height).
        tg_bbox (tuple[int], size=4): Bbox of target in tg_image, in xyxy format, endpoints inclusive.
        cs_bbox (tuple[float], size=4): Bbox of target in cropped scene image, in xyxy format.

    Returns:
        flow (np.ndarray, type=float32, size=HxWx2): A flow array with same width and height of tg_image, and two
            channels for u and v respectively. Values are in the range [0, 1] denoting coordinates relative to tg_image.
    """
    # These are floats.
    cs_x1, cs_y1, cs_x2, cs_y2 = cs_bbox
    cs_w, cs_h = cs_x2 - cs_x1, cs_y2 - cs_y1

    # These are ints.
    tg_x1, tg_y1, tg_x2, tg_y2 = tg_bbox
    tg_w, tg_h = tg_x2 - tg_x1 + 1, tg_y2 - tg_y1 + 1

    # grid contains coordinates of centers of pixels in tg_bbox, in the range [0, 1].
    grid = np.indices((tg_h, tg_w)).transpose((1, 2, 0)) + 0.5
    grid = np.flip(grid, 2)  # flip so that grid[:, :, 0] is u and grid[:, :, 1] is v
    grid /= [tg_w, tg_h]  # bring grid values in the range [0, 1]

    # Transform grid to cs_bbox via linear transformation.
    grid = grid * [cs_w, cs_h] + [cs_x1, cs_y1]

    # Place grid as part of a larger flow array.
    flow = np.zeros((tg_size[1], tg_size[0], 2), dtype=np.float32)
    flow[tg_y1:tg_y2+1, tg_x1:tg_x2+1, :] = grid

    return flow
