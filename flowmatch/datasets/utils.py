import numpy as np
from PIL import Image
import cv2

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def random_rotate_and_crop(fs_im, tg_mask, fs_bbox, theta_range, scale_factor_range):
    """Randomly rotates scene and randomly crops around fs_bbox such that object defined by tg_mask is retained in crop.

    Steps:
    1. A rotation angle (theta) is randomly sampled within theta_range. Scene is rotated by theta, and is expanded by a
    black background. An all-ones mask is also rotated by theta so that we know which pixels in the tilted scene belong
    to the original image and which ones are background.
    2. tg_mask is also rotated by theta. This is used to determine the tight, axis-aligned bounding box around
    rotated object. This bbox will be called "tfs_bbox" (axis-aligned bounding box).
    3. Call random_crop_bbox to randomly crop scene (called crop_scene) around aa_bbox.
    4. Compute rotation matrix (3x3 matrix where last row and last column is a unit vector) that transforms (u, v)
    coordinates relative to fs_bbox, to (u, v) coordinates relative to crop_scene (0 <= u, v <= 1).

    Args:
        fs_im (PIL.Image.Image): Full scene image.
        tg_mask (np.ndarray, type=np.uint8): Binary mask of target in full scene.
        fs_bbox (tuple(float), length 4): Tight target bbox in full scene, in xyxy format.
            fs_bbox is a continuous float where 0 <= x1,x2 <= width and 0 <= y1,y2 <= height.
        theta_range (tuple(float), length 2): Range (in degrees) from which theta has to be sampled uniformly.
        scale_factor_range (tuple(int), length 2): Range from which scale_factor has to be sampled randomly.

    Returns:
        tfs_im (PIL.Image.Image): Tilted full scene.
        tfs_bbox (tuple(float), length 4): Axis parallel target bbox relative to tilted full scene, in xyxy format.
            bbox is a continuous float where 0 <= x1,x2 <= 1 and 0 <= y1,y2 <= 1.
        cs_im (PIL.Image.Image): Rotated and cropped scene image.
        cs_mask (np.ndarray, type=np.uint8): Binary mask of cropped scene.
            Values in image outside mask are all zeros due to rotation.
        cs_bbox (tuple(float), length 4): Bbox of cs_im relative to tfs_im, lying in [0, 1], in xyxy format.
            bbox is a continuous float where 0 <= x1,x2 <= 1 and 0 <= y1,y2 <= 1.
        cs_rmat (np.ndarray, type=np.float32, size=(3, 3)): Rotation matrix of transformation that maps (u, v) wrt
            fs_bbox to (u, v) wrt cs_bbox.
    """
    assert (fs_im.width == tg_mask.shape[1] and fs_im.height == tg_mask.shape[0])

    # STEP 1: Rotate fs_im and tg_mask.
    theta = np.random.uniform(*theta_range)

    # Rotate fs_im : tfs_im.
    tfs_im = fs_im.rotate(theta, expand=True)  # tilted full scene

    # Create tfs_mask by rotating an all-ones array with expand=True.
    tfs_mask = np.ones((fs_im.height, fs_im.width), dtype=np.uint8)
    tfs_mask = Image.fromarray(tfs_mask).rotate(theta, expand=True)
    tfs_mask = np.array(tfs_mask)

    # Rotate tg_mask.
    tfs_tg_mask = Image.fromarray(tg_mask).rotate(theta, expand=True)
    tfs_tg_mask = np.array(tfs_tg_mask)

    # STEP 2: Compute tfs_bbox in tilted_fs_im.
    x_inds, y_inds = tfs_tg_mask.max(axis=0).nonzero()[0], tfs_tg_mask.max(axis=1).nonzero()[0]
    # While converting int coords to float coords, add 1 to higher coord.
    x1, x2, y1, y2 = float(x_inds.min()), float(x_inds.max()) + 1., float(y_inds.min()), float(y_inds.max()) + 1.
    tfs_bbox = (x1, y1, x2, y2)

    # STEP 3: Randomly crop scene around aa_bbox.
    cs_im, cs_mask, cs_bbox = random_crop_bbox(tfs_im, tfs_mask, tfs_bbox, scale_factor_range)

    # STEP 4: Compute rotation matrix that maps (u, v) wrt fs_bbox to (u, v) wrt cs_bbox in crop_scene.

    # First compute rotation matrix that maps (u, v) wrt fs_bbox (abb. as fsbb) to (x, y) wrt fs_im.
    # rmat_fsbb_uv_fsim_xy: (u, v) -> (u * fsbb_w + fsbb_x1, v * fsbb_h + fsbb_y1).
    fsbb_x1, fsbb_y1, fsbb_x2, fsbb_y2 = fs_bbox
    fsbb_w, fsbb_h = fsbb_x2 - fsbb_x1, fsbb_y2 - fsbb_y1
    rmat_fsbb_uv_fsim_xy = np.array([[fsbb_w, 0.,     fsbb_x1],
                                     [0.,     fsbb_h, fsbb_y1],
                                     [0.,     0.,     1.   ]])

    # Then compute rotation matrix that maps (x, y) wrt fs_im to (u, v) wrt tilted_fs_im.
    # The idea is that the midpoint of fs_im corresponds to the midpoint of tilted_fs_im.
    # Step 1: Center about center of fs_im: (x, y) -> (x - fs_xmid, y - fs_ymid).
    # Step 2: Rotate by an angle theta.
    # Step 3: Add center of tfs_im: (x, y) -> ((x + tfs_xmid) / tfs_w, (y + tfs_ymid) / tfs_h).
    fs_xmid, fs_ymid = fs_im.width / 2, fs_im.height / 2
    tfs_xmid, tfs_ymid = tfs_im.width / 2, tfs_im.height / 2
    rmat_step1 = np.array([[1.,   0., -fs_xmid],
                           [0.,   1., -fs_ymid],
                           [0.,   0., 1.      ]])
    angle = np.deg2rad(-theta)  # negative since y axis in image coordinates is inverted
    rmat_step2 = np.array([[np.cos(angle), -np.sin(angle),  0.],
                           [np.sin(angle),  np.cos(angle),  0.],
                           [0.,             0.,             1.]])
    rmat_step3 = np.array([[1., 0., tfs_xmid],
                           [0., 1., tfs_ymid],
                           [0., 0., 1.]])
    rmat_step3[0] /= tfs_im.width; rmat_step3[1] /=  tfs_im.height
    rmat_fsim_xy_tfsim_uv = rmat_step3 @ rmat_step2 @ rmat_step1  # multiply all matrices

    # Finally compute rotation matrix that maps (u, v) wrt tfs_im to (u, v) wrt cs_bbox.
    # rmat_tfsim_uv_csbbox_uv: (u, v) -> ((u - cs_bbox_x1) / cs_bbox_w, (v - cs_bbox_y1) / cs_bbox_h).
    cb_x1, cb_y1, cb_x2, cb_y2 = cs_bbox
    cb_w, cb_h = cb_x2 - cb_x1, cb_y2 - cb_y1
    rmat_tfsim_uv_cb_uv = np.array([[1., 0., -cb_x1],
                                    [0., 1., -cb_y1],
                                    [0., 0., 1.]])
    rmat_tfsim_uv_cb_uv[0] /= cb_w; rmat_tfsim_uv_cb_uv[1] /= cb_h

    # Compute overall rotation matrix.
    cs_rmat = rmat_tfsim_uv_cb_uv @ rmat_fsim_xy_tfsim_uv @ rmat_fsbb_uv_fsim_xy
    cs_rmat = cs_rmat.astype(np.float32)

    return tfs_im, tfs_bbox, cs_im, cs_mask, cs_bbox, cs_rmat


def random_homography_and_crop(tg_img, tg_mask, project_limit=0.4, theta_range=[-45, 45]):
    h,w = tg_img.shape[:2]

    # Pick four points for axis projection
    rect =np.float32([[0,0],[w,0],[0,h], [w,h]])
    hpos = (np.random.rand(4)-0.5) * h * project_limit
    wpos = (np.random.rand(4)-0.5) * w * project_limit
    dst = np.float32(rect + np.vstack([wpos, hpos]).T)

    # Random rotation
    theta = np.random.uniform(*theta_range)
    theta = np.deg2rad(theta)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0,0,1]])
    dst = np.hstack([dst, np.ones([4,1])]).T
    dst = rot.dot(dst).T[:,:2].astype(np.float32)

    # Resize the transformed object image and paste it onto a canvas of 
    # original size
    canvas_h, canvas_w = np.max(dst[:,0])-np.min(dst[:,0]), np.max(dst[:,1])-np.min(dst[:,1])
    dx, dy = -np.min(dst[:,0]), -np.min(dst[:,1])
    M = cv2.getPerspectiveTransform(rect, dst)
    pts = np.float32([[0,0,1],[w,0,1],[0,h,1], [w,h,1]]).T
    scale = np.array([[w/canvas_h, 0, 0],[0,h/canvas_w,0], [0,0,1]])
    trans = np.array([[1,0,dx], [0,1,dy],[0,0,1]])
    M = scale.dot(trans).dot(M)
    sc_img = cv2.warpPerspective(tg_img, M, (w,h))
    sc_mask = cv2.warpPerspective(tg_mask, M, (w,h))
    return sc_img, sc_mask, M


def random_crop_bbox(fs_im, fs_mask, bbox, scale_factor_range):
    """Randomly crops scene around bbox such that bbox lies completely inside crop.
    Steps:
    1. A scale_factor is randomly sampled within scale_factor_range. "crop_box" is defined as a *square* box that is
    scale_factor times larger than the original bbox. scale_factor will be truncated so that crop_box lies inside full
    scene.
    2. The ranges of x1 and y1 of the crop_box along x and y dimension is ascertained, based on the crop_box completely
    containing the bbox and completely lying inside the full scene.
    3. x1 and y1 are sampled from these ranges. This defines a unique crop_box, which is cropped.
    Args:
        fs_im (PIL.Image.Image): Full scene image.
        fs_mask (np.ndarray, type=np.uint8): Binary mask of full scene image. When fs_im is cropped, fs_mask should also
            be correspondingly cropped.
        bbox (tuple(float), length 4): Tight target bbox, relative to full scene, in xyxy format.
            bbox is a continuous float where 0 <= x1,x2 <= width and 0 <= y1,y2 <= height.
        scale_factor_range (tuple(int), length 2): Range from which scale_factor has to be sampled randomly.
    Returns:
        crop_scene (PIL.Image.Image): cropped scene image.
        cs_bbox (tuple(float), length 4): Bounding box of crop_scene relative to full_scene, in xyxy format.
            cs_bbox is a continuous float where 0 <= x1,x2 <= 1 and 0 <= y1,y2 <= 1.
    """
    # Convert bbox floats to ints.
    bb_x1, bb_y1, bb_x2, bb_y2 = bbox
    # When going from continuous float coordinates to pixel index coordinates, need to subtract 1 from x2 and y2.
    bb_x1, bb_y1, bb_x2, bb_y2 = int(bb_x1), int(bb_y1), int(bb_x2) - 1, int(bb_y2) - 1
    bb_w, bb_h = bb_x2 - bb_x1 + 1, bb_y2 - bb_y1 + 1

    # Calculate crop box size.
    scale_factor = np.random.uniform(*scale_factor_range)
    cb_size = int(scale_factor * max(bb_w, bb_h))  # cb is an acronym for crop_box
    cb_size = min(cb_size, fs_im.width, fs_im.height)

    # Randomly compute x1 and x2 of crop_box.
    # There are two constraints, 0 <= x1, x2 < width, x1 <= bbox_x1, bbox_x2 <= x2.
    cb_x1_min, cb_x1_max = max(0, bb_x2 - cb_size + 1), min(fs_im.width - cb_size, bb_x1)
    if (cb_x1_min > cb_x1_max):
        print('fs_im.size - {}'.format(fs_im.size))
        print("bb_x1, bb_y1, bb_x2, bb_y2 - ({}, {}, {}, {})".format(bb_x1, bb_y1, bb_x2, bb_y2))
        print("bb_w, bb_h - ({}, {})".format(bb_w, bb_h))
        print("scale factor - {}".format(scale_factor))
        print("cb_size - {}".format(cb_size))
        print("cb_x1_min, cb_x1_max - ({}, {})".format(cb_x1_min, cb_x1_max))
        raise Exception("cb_x1_min greater than cb_x1_max")
    cb_x1 = np.random.randint(cb_x1_min, cb_x1_max + 1)
    cb_x2 = cb_x1 + cb_size - 1

    # Randomly compute y1 and y2 of crop_box.
    # There are two constraints, 0 <= y1, y2 < height, y1 <= bbox_y1, bbox_y2 <= y2.
    cb_y1_min, cb_y1_max = max(0, bb_y2 - cb_size + 1), min(fs_im.height - cb_size, bb_y1)
    if (cb_y1_min > cb_y1_max):
        print('fs_im.size - {}'.format(fs_im.size))
        print("bb_x1, bb_y1, bb_x2, bb_y2 - ({}, {}, {}, {})".format(bb_x1, bb_y1, bb_x2, bb_y2))
        print("bb_w, bb_h - ({}, {})".format(bb_w, bb_h))
        print("scale factor - {}".format(scale_factor))
        print("cb_size - {}".format(cb_size))
        print("cb_y1_min, cb_y1_max - ({}, {})".format(cb_y1_min, cb_y1_max))
        raise Exception("cb_y1_min greater than cb_y1_max")
    cb_y1 = np.random.randint(cb_y1_min, cb_y1_max + 1)
    cb_y2 = cb_y1 + cb_size - 1

    # Get cropped scene image.
    cs_arr = np.array(fs_im)[cb_y1:cb_y2 + 1, cb_x1:cb_x2 + 1, :]
    cs_im = Image.fromarray(cs_arr)

    # Get cropped mask.
    cs_mask = fs_mask[cb_y1:cb_y2 + 1, cb_x1:cb_x2 + 1]

    # Get crop_bbox w.r.t. full_scene.
    cb_x1, cb_y1 = cb_x1 / fs_im.width, cb_y1 / fs_im.height
    # While converting int coords to continuous float coords, 1 has to be added to the higher coord.
    cb_x2, cb_y2 = (cb_x2 + 1) / fs_im.width, (cb_y2 + 1) / fs_im.height
    cs_bbox = (cb_x1, cb_y1, cb_x2, cb_y2)

    return cs_im, cs_mask, cs_bbox


def center_crop(fs_im, bbox, scale_factor=1.2):
    # Convert bbox floats to ints.
    bb_x1, bb_y1, bb_x2, bb_y2 = bbox
    # When going from continuous float coordinates to pixel index coordinates, need to subtract 1 from x2 and y2.
    bb_x1, bb_y1, bb_x2, bb_y2 = int(bb_x1), int(bb_y1), int(bb_x2) - 1, int(bb_y2) - 1
    bb_w, bb_h = bb_x2 - bb_x1 + 1, bb_y2 - bb_y1 + 1
    # Calculate crop box size.
    cb_size = int(scale_factor * max(bb_w, bb_h))  # cb is an acronym for crop_box
    cs_size = min(cb_size, fs_im.width, fs_im.height)

    # Compute x1 and x2 of crop_box.
    xmid, ymid = (bb_x1 + bb_x2) // 2, (bb_y1 + bb_y2) // 2
    cs_x1, cs_y1 = max(0, xmid - cs_size // 2), max(0, ymid - cs_size // 2)
    cs_x2, cs_y2 = cs_x1 + cs_size - 1, cs_y1 + cs_size - 1
    cs_x2, cs_y2 = min(cs_x2, fs_im.width - 1), min(cs_y2, fs_im.height - 1)
    cs_x1, cs_y1 = cs_x2 - cs_size + 1, cs_y2 - cs_size + 1

    # Get cropped scene image.
    cs_arr = np.array(fs_im)[cs_y1:cs_y2+1, cs_x1:cs_x2+1, :]
    cs_im = Image.fromarray(cs_arr)

    # Get crop_bbox w.r.t. full_scene.
    fs_cs_xy_bbox = (cs_x1, cs_y1, cs_x2, cs_y2)

    return cs_im, fs_cs_xy_bbox

def xywh2xyxy(bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    bbox = x1, y1, x2, y2
    return bbox

def id_to_str(id, block):
    """Converts a tuple-based id to string
    Args:
        id (tuple(int)): id which is a tuple of ints.
        block: Length of each element of tuple when converted to string. Will be filled with leading zeros.

    >>> id_to_str((3, 14, 5), block=6)
    '000003000014000005'
    """
    if max(id) >= 10**(block+1):
        raise Exception('Id element {} length greater than block={}'.format(max(id), block))
    return ''.join([str(e).zfill(block) for e in id])


def str_to_id(string, block):
    """Converts a string to a tuple-based id
    Args:
        string (str): Length is N*block, where each block is an int with leading zeros.
        block: Length of each block. Each block is an int with leading zeros.

    >>> str_to_id('000003000014000005', block=6)
    (3, 14, 5)
    """
    if len(string) % block != 0:
        raise Exception('String length not a multiple of block={}'.format(block))
    num_blocks = len(string) // block
    return tuple([int(string[i*block: (i+1)*block]) for i in range(num_blocks)])
