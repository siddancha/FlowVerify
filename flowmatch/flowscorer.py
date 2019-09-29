import cv2
import numpy as np
from flowmatch.flowutils.vis import warp, split_tg_mask_wrt_bbox, get_flow_codomain_mask_gradient_based


class FlowMatchScorer:
    score_names = 'ncc_norm', 'ncc_stddev', 'f_inlier', 'fp', 'fr', 'ff1'

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

        dot = self._rgbvec_dot(norm_tg_rgbvec, norm_wp_rgbvec)
        return dot

    def _ncc_with_stddev(self):
        # Normalize vectors.
        norm_tg_rgbvec = self._normalize_rgb_vector(self.tg_rgbvec, np.std)
        norm_wp_rgbvec = self._normalize_rgb_vector(self.wp_rgbvec, np.std)

        dot = self._rgbvec_dot(norm_tg_rgbvec, norm_wp_rgbvec)
        return dot

    def _orb(self):
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(self.tg_arr[:, :, ::-1].copy(), None)
        kp2, des2 = orb.detectAndCompute(self.wp_arr[:, :, ::-1].copy(), None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)

        return len(matches)

    def _sift(self):
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.tg_arr[:, :, ::-1].copy(), None)
        kp2, des2 = sift.detectAndCompute(self.wp_arr[:, :, ::-1].copy(), None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        return len(good)

    def _fp(self):
        flow_in_tg_mask, flow_out_tg_mask = split_tg_mask_wrt_bbox(self.flow, self.mask, self.bbox)
        self.example['flow_in_tg_mask'], self.example['flow_out_tg_mask'] = flow_in_tg_mask, flow_out_tg_mask
        self.fp_score = flow_in_tg_mask.sum() / self.mask.sum()
        return self.fp_score

    def _fr(self):
        cd_mask = get_flow_codomain_mask_gradient_based(self.flow, self.mask)

        bbox_mask = np.zeros_like(cd_mask)
        cd_mask_side = cd_mask.shape[0]
        x1, y1, x2, y2 = (int(cd_mask_side * e) for e in self.bbox)
        bbox_mask[y1:y2, x1:x2] = 1

        cd_mask_in = cd_mask * bbox_mask
        cd_mask_out = cd_mask * (1 - bbox_mask)

        self.example['codomain_mask'] = cd_mask
        self.example['codomain_in_mask'] = cd_mask_in
        self.example['codomain_out_mask'] = cd_mask_out

        self.fr_score = cd_mask_in.sum() / bbox_mask.sum()
        return self.fr_score

    def _ff1(self):
        if not hasattr(self, 'fp_score'):
            self._fp()
        if not hasattr(self, 'fr_score'):
            self._fr()

        f1_score = (2 * self.fp_score * self.fr_score) / (self.fp_score + self.fr_score)
        return f1_score

    def _f_inlier(self, width=256, height=256):
        grid = np.indices((height, width)).transpose((1, 2, 0))
        grid = np.flip(grid, 2)  # flip so that grid[:, :, 0] is u and grid[:, :, 1] is v
        grid = grid[self.y_inds, self.x_inds]
        flow = self.flow * [width, height]
        flow = flow[self.y_inds, self.x_inds]
        F, mask = cv2.findFundamentalMat(grid, flow, cv2.FM_RANSAC, 3, 0.9)
        num_inlier = np.sum(mask)
        return num_inlier / self.tg_rgbvec.shape[0]

    def get_scores(self):
        return self._ncc_with_norm(), self._ncc_with_stddev(), self._f_inlier(), self._fp(), self._fr(), self._ff1()

    # # Jenny's code
    # def get_score_map(self, sz=256):
    #     score_map = np.zeros((sz, sz))
    #     score_map[self.y_inds, self.x_inds] = self.score_vec
    #     norm_tg = self._normalize_rgb_vector(self.tg_rgbvec, np.linalg.norm)
    #     tg_map = np.zeros((sz, sz, 3))
    #     tg_map[self.y_inds, self.x_inds] = norm_tg
    #     norm_wp_rgbvec = self._normalize_rgb_vector(self.wp_rgbvec, np.linalg.norm)
    #     wp_map = np.zeros((sz, sz, 3))
    #     wp_map[self.y_inds, self.x_inds] = norm_wp_rgbvec
    #     return tg_map * 255, wp_map * 255, score_map * 255, self.edge_map, np.array(self.mask_wp) * 255
