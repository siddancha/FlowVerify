import cv2
import numpy as np

from PIL import Image
from flowmatch.flowutils.computeColor import computeImg


def warp(flow, cs_im, tg_mask):
    """Warps cropped scene image based on flow.

    Since we only care about warping to target image within the mask, all points outside the mask are zeroed.

    Args:
        flow (np.ndarray, size=(N,N,2), type=np.float32): Flow field with values in range [0, 1].
        cs_im (PIL.Image.Image): Cropped scene image being warped to target image.
        tg_mask (np.ndarray, size=(N, N), type=np.uint8): Binary mask of target image.
    Returns:
        warp_im (PIL.Image.Image): Warped image.
    """
    # Rescale flow values from [0, 1] to [0, width/height].
    u = flow[:, :, 0] * cs_im.width
    v = flow[:, :, 1] * cs_im.height

    cs_arr = np.array(cs_im)
    warp_arr = cv2.remap(cs_arr, u, v, interpolation=cv2.INTER_LINEAR)

    # Zero out warped image that lies outside tg_mask.
    y_inds, x_inds = (tg_mask == 0).nonzero()  # indices where mask is 0
    warp_arr[y_inds, x_inds] = 0

    # Convert array to PIL.Image.Image.
    warp_im = Image.fromarray(warp_arr)

    return warp_im


def get_color_coding(size):
    """Computes color coding for a square image.
    Args:
        size (int): Size of the NxN image.
    Returns:
        cc_img (np.ndarray, size=sizexsizex3, type=np.uint8): Color coding image.
    """
    flow = np.indices((size, size)).transpose(1, 2, 0)
    flow = np.flip(flow, 2)
    flow = flow - size / 2
    cc_arr = computeImg(flow)
    return cc_arr


def split_tg_mask_wrt_bbox(flow, tg_mask, bbox):
    """
    Splits indices in tg_mask into those that map inside bbox and those that map outside (mapping is flow).
    :param flow: (np.ndarray(dtype=float, size=HxHx2)) flow field.
    :param tg_mask: (np.ndarray(dtype=uint8, size=HxH)) mask of target.
    :param bbox: (tuple(float, float, float, float)) bbox in relative coordinates (lies between 0 and 1).
    :return: (tuple(np.ndarray(dtype=uint8, size=HxH), len=2)) masks for points mapping to inside and outside bbox.
    """
    u1, v1, u2, v2 = bbox
    inside_bbox = np.logical_and.reduce([u1 <= flow[:, :, 0],
                                         u2 >= flow[:, :, 0],
                                         v1 <= flow[:, :, 1],
                                         v2 >= flow[:, :, 1]]).astype(np.uint8)
    inside_bbox_tg_mask = tg_mask * inside_bbox
    outside_bbox_tg_mask = tg_mask * (1 - inside_bbox)
    return inside_bbox_tg_mask, outside_bbox_tg_mask


def get_flow_codomain_mask_quadr_mapping(flow, tg_mask, resf=8):
    """
    Computes mask which contains points that are mapped to by flow, via mapping four pixels at a time.
    :param flow: (np.ndarray{dtype=float, size=HxHx2}) flow field.
    :param tg_mask: (np.ndarray{dtype=uint8, size=HxH}) mask of target.
    :param resf: (float) resolution factor; returned map's resolution will resf times.
    :return: cd_mask: (np.ndarray{dtype=uint8, size=resf*H x resf*H}) codomain mask.
    """
    tg_mask_side = tg_mask.shape[0]
    cd_mask_side = resf * tg_mask_side  # cd is abbreviation for codomain
    cd_mask = np.zeros([cd_mask_side, cd_mask_side], dtype=np.uint8)

    # Scale flow by length of cd_mask. These are the positions being mapped to in cd_mask.
    cd_xy = np.clip((flow * cd_mask_side).astype(np.int32), a_min=0, a_max=cd_mask_side-1)

    for tg_y1, tg_x1 in zip(*np.where(tg_mask)):
        tg_x2, tg_y2 = tg_x1 + 1, tg_y1 + 1
        tg_p1, tg_p2, tg_p3, tg_p4 = (tg_x1, tg_y1), (tg_x1, tg_y2), (tg_x2, tg_y2), (tg_x2, tg_y1)
        quadr = np.array([cd_xy[tg_y, tg_x] for tg_x, tg_y in (tg_p1, tg_p2, tg_p3, tg_p4)], dtype=np.int32)
        cv2.fillConvexPoly(cd_mask, quadr, 1)

    return cd_mask


def get_flow_codomain_mask_gradient_based(flow, tg_mask, resf=8):
    """
    Computes mask which contains points that are mapped to by flow.
    :param flow: (np.ndarray{dtype=float, size=HxHx2}) flow field.
    :param tg_mask: (np.ndarray{dtype=uint8, size=HxH}) mask of target.
    :param resf: (float) resolution factor; returned map's resolution will resf times.
    :return: cd_mask: (np.ndarray{dtype=uint8, size=resf*H x resf*H}) codomain mask.
    """
    tg_mask_side = tg_mask.shape[0]
    cd_mask_side = resf * tg_mask_side  # cd is abbreviation for codomain
    cd_mask = np.zeros([cd_mask_side, cd_mask_side], dtype=np.uint8)

    # Scale flow by length of cd_mask. These are the positions being mapped to in cd_mask.
    xy = flow * cd_mask_side

    # Compute flow derivatives.
    dxy_1 = np.zeros_like(flow)
    dxy_1[:, 1:-1, :] = (xy[:, 2:, :] - xy[:, :-2, :]) / 2.
    dxy_1[:, 0, :] = xy[:, 1, :] - xy[:, 0, :]
    dxy_1[:, -1, :] = xy[:, -1, :] - xy[:, -2, :]

    dxy_2 = np.zeros_like(flow)
    dxy_2[1:-1, :, :] = (xy[2:, :, :] - xy[:-2, :, :]) / 2.
    dxy_2[0, :, :] = xy[1, :, :] - xy[0, :, :]
    dxy_2[-1, :, :] = xy[-1:, :, :] - xy[-2, :, :]

    xy1 = xy - dxy_1 / 2 - dxy_2 / 2
    xy2 = xy - dxy_1 / 2 + dxy_2 / 2
    xy3 = xy + dxy_1 / 2 + dxy_2 / 2
    xy4 = xy + dxy_1 / 2 - dxy_2 / 2

    # Select parallelograms for points inside tg_mask.
    tg_mask_ys, tg_mask_xs = np.where(tg_mask)
    xy1, xy2, xy3, xy4 = (xyi[tg_mask_ys, tg_mask_xs, :].astype(np.int32) for xyi in (xy1, xy2, xy3, xy4))
    pllgrams = np.stack((xy1, xy2, xy3, xy4), axis=1)

    for pllgram in pllgrams:
        cv2.fillConvexPoly(cd_mask, pllgram, 1)

    return cd_mask


def overlay_color_coding(bg_im, alphas=(0.2, 0.8)):
    """Overlay color coding on image.

    Image has to be square.

    Args:
        bg_im (PIL.Image.Image): Background image.
    Returns:
        blended_im (PIL.Image.Image): Background image blended with color coding.
        overlay_im (PIL.Image.Image): Overlay image.
    """
    assert(bg_im.width == bg_im.height)  # image has to be square
    size = bg_im.width

    zero_im = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
    blended_im = Image.blend(bg_im, zero_im, alpha=alphas[0])

    cc_arr = get_color_coding(size)
    overlay_im = Image.fromarray(cc_arr)
    blended_im = Image.blend(blended_im, overlay_im, alpha=alphas[1])

    return blended_im, overlay_im
