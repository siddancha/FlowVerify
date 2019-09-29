import cv2
import numpy as np
from flowmatch.flowutils.vis import warp, split_tg_mask_wrt_bbox, get_flow_codomain_mask_gradient_based
from pipeline.eval.utils import compute_iou

class FlowMatchScorer:
    score_names = 'FRigid', 'FColor', 'FPrec', 'FRec'
    
    def __init__(self, example):
        self.example = example

        cs_im = example['cs_im']
        resized_tg_arr, resized_tg_mask = np.array(example['resized_tg_im']), example['resized_tg_mask']
        self.flow = example['pred_flow']

        # Warp cropped scene using flow.
        warped_im = warp(self.flow, cs_im, resized_tg_mask)
        warped_arr = np.array(warped_im)
        example['warped_im'] = warped_im

        self.tg_arr, self.wp_arr, self.mask = resized_tg_arr, warped_arr, resized_tg_mask

        # Select points inside masks: shape from (N, N, 3) to (M, 3)
        self.y_inds, self.x_inds = np.where(self.mask)
        self.tg_rgbvec = self.tg_arr[self.y_inds, self.x_inds]
        self.wp_rgbvec = self.wp_arr[self.y_inds, self.x_inds]

        self.bbox = example['cs_dt_uv_bbox']

    def _normalize_rgb_vector(self, rgbvec, denom_fn):
        rgbvec = rgbvec - np.mean(rgbvec, axis=0)
        rgbvec = rgbvec / denom_fn(rgbvec, axis=0)
        return rgbvec

    def _rgbvec_dot(self, rgbvec1, rgbvec2):
        vec1, vec2 = rgbvec1.flatten(), rgbvec2.flatten()
        return np.dot(vec1, vec2) / 3

    def _ncc_with_norm(self):
        # Normalize vectors.
        norm_tg_rgbvec = self._normalize_rgb_vector(self.tg_rgbvec, np.linalg.norm)
        norm_wp_rgbvec = self._normalize_rgb_vector(self.wp_rgbvec, np.linalg.norm)

        dot = 0.5*(1+self._rgbvec_dot(norm_tg_rgbvec, norm_wp_rgbvec))
        return dot

    def _fp(self):
        flow_in_tg_mask, flow_out_tg_mask = split_tg_mask_wrt_bbox(self.flow, self.mask, self.bbox)
        self.example['flow_in_tg_mask'], self.example['flow_out_tg_mask'] = flow_in_tg_mask, flow_out_tg_mask
        self.fp_score = flow_in_tg_mask.sum() / self.mask.sum()
        return self.fp_score
 
    def _fr(self):
        u = self.flow[:, :, 0] * self.mask.shape[1] #width
        v = self.flow[:, :, 1] * self.mask.shape[0]
        u = u[self.y_inds, self.x_inds]
        v = v[self.y_inds, self.x_inds]

        x1, x2 = np.min(u), np.max(u)
        y1, y2 = np.min(v), np.max(v)
        u1, v1, u2, v2 = self.example['cs_dt_uv_bbox']
        cd_mask_side = self.mask.shape[0]
        xb1, yb1, xb2, yb2 = u1 * cd_mask_side, v1 * cd_mask_side, u2 * cd_mask_side, v2 * cd_mask_side

        self.fr_score = compute_iou([x1, y1, x2, y2], [xb1, yb1, xb2, yb2])
        self.example['fr_bbox'] = [x1/ cd_mask_side, y1/ cd_mask_side, 
                                    x2/ cd_mask_side, y2/ cd_mask_side]
        return self.fr_score

    def _f_inlier(self, width=256, height=256):
        grid = np.indices((height, width)).transpose((1, 2, 0))
        grid = np.flip(grid, 2)
        grid = grid[self.y_inds, self.x_inds]
        flow = self.flow * [width, height]
        flow = flow[self.y_inds, self.x_inds]
        F, mask = cv2.findFundamentalMat(grid, flow, cv2.FM_RANSAC, 8, 0.9)
        num_inlier = np.sum(mask)
        return num_inlier / self.tg_rgbvec.shape[0]

    def get_scores(self):
        return self._f_inlier(), self._ncc_with_norm(), self._fp(), self._fr() 
