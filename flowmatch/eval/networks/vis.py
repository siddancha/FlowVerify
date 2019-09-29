import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, patches
import numpy as np

from flowmatch.flowutils.computeColor import computeImg
from flowmatch.flowutils.vis import warp


def _blend_images(list_arr, list_alphas):
    alphas = np.array(list_alphas).astype(np.float32)
    blended_arr = list_arr[0]
    for i in range(1, len(list_arr)):
        a1, a2 = alphas[:i].sum(), alphas[i]
        a1, a2 = a1 / (a1 + a2), a2 / (a1 + a2)
        blended_arr = cv2.addWeighted(blended_arr, a1, list_arr[i], a2, 0)
    return blended_arr


def _draw_bbox(ax, bbox, color):
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def _draw_full_scene(ax, example):
    fs_arr, fs_cs_xy_bbox, fs_dt_xy_bbox, fs_gt_xy_bbox =\
        np.array(example['fs_im']), example['fs_cs_xy_bbox'], example['fs_dt_xy_bbox'], example['fs_gt_xy_bbox']
    ax.imshow(fs_arr)
    _draw_bbox(ax, fs_cs_xy_bbox, color='r')
    _draw_bbox(ax, fs_dt_xy_bbox, color='b')
    if fs_gt_xy_bbox is not None:
        _draw_bbox(ax, fs_gt_xy_bbox, color='g')
    ax.set_xticks([]); ax.set_yticks([])


def _draw_cropped_scene(ax, example):
    cs_arr, tg_arr = np.array(example['cs_im']), np.array(example['resized_tg_im'])
    max_size = max(cs_arr.shape[0], tg_arr.shape[0])
    if cs_arr.shape[0] != max_size:
        cs_arr = cv2.resize(cs_arr, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
    ax.imshow(cs_arr)
    if 'cs_dt_xy_bbox' in example.keys():
        cs_dt_xy_bbox = example['cs_dt_xy_bbox']
        _draw_bbox(ax, cs_dt_xy_bbox, color='r')
    ax.set_xticks([]); ax.set_yticks([])


def _draw_target_image(ax, example):
    ax.imshow(example['resized_tg_im'])
    ax.set_xticks([]); ax.set_yticks([])


def _draw_flow(ax, example, flow):
    centered_flow = flow - 0.5
    centered_flow *= example['resized_tg_mask'][:, :, None]
    ax.imshow(computeImg(centered_flow))
    ax.set_xticks([]); ax.set_yticks([])


def _draw_flow_diff(fig, ax1, ax2, example, flow1, flow2):
    heatmap = np.linalg.norm(flow1 - flow2, axis=2)
    heatmap *= example['resized_tg_mask'][:, :, None]
    hm1 = ax1.imshow(heatmap, cmap='binary', interpolation='nearest')
    fig.colorbar(hm1, ax=ax1)
    hm2 = ax2.imshow(heatmap, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    fig.colorbar(hm2, ax=ax2)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2.set_xticks([]); ax2.set_yticks([])


def _draw_flow_arrow_image_for_sift(ax, example, num_arrows=10):
    cs_arr, tg_arr, tg_mask = np.array(example['cs_im']), np.array(example['resized_tg_im']), example['resized_tg_mask']

    kp_cs, kp_tg = example['kp_cs'], example['kp_tg']
    matches = example['pred_sift']

    cs_w, cs_h = example['resized_cs_im'].size
    tg_w, tg_h = example['resized_tg_im'].size

    list_kpuv_cs = [(kp.pt[0] / cs_w, kp.pt[1] / cs_h) for kp in kp_cs]
    list_kpuv_tg = [(kp.pt[0] / tg_w, kp.pt[1] / tg_h) for kp in kp_tg]

    assert (cs_arr.shape[0] == cs_arr.shape[1])
    assert (tg_arr.shape[0] == tg_arr.shape[1])
    assert (tg_arr.shape[:2] == tg_mask.shape)

    max_size = max(cs_arr.shape[0], tg_arr.shape[0])
    if cs_arr.shape[0] != max_size:
        cs_arr = cv2.resize(cs_arr, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
    if tg_arr.shape[0] != max_size:
        tg_arr = cv2.resize(tg_arr, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
        # tg_mask = cv2.resize(tg_mask, (max_size, max_size), interpolation=cv2.INTER_NEAREST)

    gap_size = int(.05 * max_size)
    gap_arr = 255 * np.ones((max_size, gap_size, 3), dtype=np.uint8)
    new_arr = np.concatenate([tg_arr, gap_arr, cs_arr], axis=1)
    ax.imshow(new_arr)

    num_arrows = min(len(matches), num_arrows)

    if num_arrows > 0:
        cmap = cm.get_cmap('tab10')
        step = 1. / num_arrows
        colors = [cmap(e + 0.5 * step) for e in np.arange(0., 1., step)]
        scatter_size, head_width = 3e-2 * max_size, 1e-2 * max_size

        matches = np.random.choice(matches, size=num_arrows, replace=False)
        for i, match in enumerate(matches):
            color = colors[i]
            u1, v1 = list_kpuv_tg[match.queryIdx]
            u2, v2 = list_kpuv_cs[match.trainIdx]
            x1, y1, x2, y2 = [e * max_size for e in [u1, v1, u2, v2]]

            # Point in image 1.
            ax.scatter([x1], [y1], color=color, alpha=.5, s=scatter_size)

            if 0 <= x2 <= max_size and 0 <= y2 <= max_size:
                x2 = x2 + max_size + gap_size  # x2 in image 2 is shifted by max_size + gap_size

                # Point in image 2.
                ax.scatter([x2], [y2], color=color, alpha=.5, s=scatter_size)

                # Arrow from image 1 to image 2.
                ax.arrow(x1, y1, x2 - x1, y2 - y1, color=color, length_includes_head=True, head_width=head_width)

    ax.set_xticks([]); ax.set_yticks([])


def _draw_flow_arrow_image(ax, example, flow, num_arrows=10):
    cs_arr, tg_arr, tg_mask = np.array(example['cs_im']), np.array(example['resized_tg_im']), example['resized_tg_mask']

    assert (cs_arr.shape[0] == cs_arr.shape[1])
    assert (tg_arr.shape[0] == tg_arr.shape[1])
    assert (tg_arr.shape[:2] == tg_mask.shape == flow.shape[:2])

    max_size = max(cs_arr.shape[0], tg_arr.shape[0])
    if cs_arr.shape[0] != max_size:
        cs_arr = cv2.resize(cs_arr, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
    if tg_arr.shape[0] != max_size:
        tg_arr = cv2.resize(tg_arr, (max_size, max_size), interpolation=cv2.INTER_LINEAR)
        tg_mask = cv2.resize(tg_mask, (max_size, max_size), interpolation=cv2.INTER_NEAREST)
        flow = cv2.resize(flow, (max_size, max_size), interpolation=cv2.INTER_LINEAR)

    red_vec, blue_vec = np.array([[[255, 0, 0]]], dtype=np.uint8), np.array([[[0, 0, 255]]], dtype=np.uint8)
    alphas = [0.6, 0.2, 0.2]

    # Add flow target masks.
    if 'flow_in_tg_mask' in example:
        mask_in = example['flow_in_tg_mask'][:, :, None] * blue_vec
        if mask_in.shape[0] != max_size:
            mask_in = cv2.resize(mask_in, (max_size, max_size), interpolation=cv2.INTER_NEAREST)

        mask_out = example['flow_out_tg_mask'][:, :, None] * red_vec
        if mask_out.shape[0] != max_size:
            mask_out = cv2.resize(mask_out, (max_size, max_size), interpolation=cv2.INTER_NEAREST)

        tg_arr = _blend_images([tg_arr, mask_in, mask_out], alphas)

    # Add codomain masks.
    if 'codomain_mask' in example:
        mask_in = example['codomain_in_mask'][:, :, None] * blue_vec
        if mask_in.shape[0] != max_size:
            mask_in = cv2.resize(mask_in, (max_size, max_size), interpolation=cv2.INTER_NEAREST)

        mask_out = example['codomain_out_mask'][:, :, None] * red_vec
        if mask_out.shape[0] != max_size:
            mask_out = cv2.resize(mask_out, (max_size, max_size), interpolation=cv2.INTER_NEAREST)

        cs_arr = _blend_images([cs_arr, mask_in, mask_out], alphas)

    gap_size = int(.05 * max_size)
    gap_arr = 255 * np.ones((max_size, gap_size, 3), dtype=np.uint8)
    new_arr = np.concatenate([tg_arr, gap_arr, cs_arr], axis=1)
    ax.imshow(new_arr)

    # Get coordinates in image 1.
    y1s, x1s = np.where(tg_mask)
    if len(x1s) > num_arrows:
        keep = np.random.choice(len(x1s), size=num_arrows, replace=False)
        x1s, y1s = x1s[keep], y1s[keep]

    # Get coordinates in image 2.
    coords2 = flow[y1s, x1s] * max_size
    x2s, y2s = coords2[:, 0], coords2[:, 1]
    x1s, y1s = x1s + 0.5, y1s + 0.5

    cmap = cm.get_cmap('tab10')
    step = 1. / num_arrows
    colors = [cmap(e + 0.5 * step) for e in np.arange(0., 1., step)]
    scatter_size, head_width = 3e-2 * max_size, 1e-2 * max_size
    for i in range(num_arrows):
        x1, y1, x2, y2, color = x1s[i], y1s[i], x2s[i], y2s[i], colors[i]

        # Point in image 1.
        ax.scatter([x1], [y1], color=color, alpha=.5, s=scatter_size)

        if 0 <= x2 <= max_size and 0 <= y2 <= max_size:
            x2 = x2 + max_size + gap_size  # x2 in image 2 is shifted by max_size + gap_size

            # Point in image 2.
            ax.scatter([x2], [y2], color=color, alpha=.5, s=scatter_size)

            # Arrow from image 1 to image 2.
            ax.arrow(x1, y1, x2 - x1, y2 - y1, color=color, length_includes_head=True, head_width=head_width)

    ax.set_xticks([]); ax.set_yticks([])


def _draw_warped_image(ax, example, flow):
    if 'warped_im' in example:
        warped_im = example['warped_im']
    else:
        warped_im = warp(flow, example['cs_im'], example['resized_tg_mask'])
    ax.imshow(warped_im)
    ax.set_xticks([]); ax.set_yticks([])


def draw_image_with_gt_flow(example):
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 12)

    # Full scene.
    ax = plt.subplot(gs[0, 0:4])
    _draw_full_scene(ax, example)

    # Cropped scene.
    ax = plt.subplot(gs[0, 4:8])
    _draw_cropped_scene(ax, example)

    # Target image.
    ax = plt.subplot(gs[0, 8:12])
    _draw_target_image(ax, example)

    # Arrows going from target image to cropped scene.
    ax = plt.subplot(gs[1, 0:6])
    _draw_flow_arrow_image(ax, example, example['pred_flow'])

    # Predicted flow.
    ax = plt.subplot(gs[1, 6:9])
    _draw_flow(ax, example, example['pred_flow'])

    # GT flow
    ax = plt.subplot(gs[1, 9:12])
    _draw_flow(ax, example, example['flow'])

    # Difference between pred and GT flows (2 images).
    ax1 = plt.subplot(gs[2, 0:3])
    ax2 = plt.subplot(gs[2, 3:6])
    _draw_flow_diff(fig, ax1, ax2, example, example['pred_flow'], example['flow'])

    # Warped image using pred flow.
    ax = plt.subplot(gs[2, 6:9])
    _draw_warped_image(ax, example, example['pred_flow'])

    # Warped image using GT flow.
    ax = plt.subplot(gs[2, 9:12])
    _draw_warped_image(ax, example, example['flow'])

    return gs, fig


def draw_image_without_gt_flow(example):
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3)

    # Full scene.
    ax = plt.subplot(gs[0, 0])
    _draw_full_scene(ax, example)

    # Cropped scene.
    ax = plt.subplot(gs[0, 1])
    _draw_cropped_scene(ax, example)

    # Target image.
    ax = plt.subplot(gs[0, 2])
    _draw_target_image(ax, example)

    # Arrows going from target image to cropped scene.
    ax = plt.subplot(gs[1, :2])
    _draw_flow_arrow_image(ax, example, example['pred_flow'])

    # Predicted flow.
    ax = plt.subplot(gs[1, 2])
    _draw_flow(ax, example, example['pred_flow'])

    # Warped image using pred flow.
    ax = plt.subplot(gs[2, 0])
    _draw_warped_image(ax, example, example['pred_flow'])

    return gs, fig


def draw_images(example, out_file, title=None):
    if 'flow' in example.keys():
        gs, fig = draw_image_with_gt_flow(example)
    else:
        gs, fig = draw_image_without_gt_flow(example)

    if title is not None:
        fig.suptitle(title, size='small')
        gs.tight_layout(fig, rect=[None, None, None, 0.99])
    else:
        gs.tight_layout()

    with open(out_file, 'wb') as f:
        plt.savefig(f, format='pdf', dpi=1200)
        plt.clf()
