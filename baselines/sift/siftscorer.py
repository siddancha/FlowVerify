import cv2
import numpy as np


class SiftMatchScorer:
    # score_names = 'sift_matches', 'sift_inliers'
    score_names = 'SMatches', 'SPrecision'

    def __init__(self, example):
        self.example = example
        self.sift_matches = example['pred_sift']

        resized_tg_arr, resized_tg_mask = np.array(example['resized_tg_im']), example['resized_tg_mask']
        self.tg_arr, self.tg_mask = resized_tg_arr, resized_tg_mask

        self.kp_cs, self.kp_tg = example['kp_cs'], example['kp_tg']

    def _sm(self):
        # """Ratio of sift matches wrt total pixels in target mask"""
        # ratio = len(self.sift_matches) / self.tg_mask.sum()
        # assert(0. <= ratio <= 1.)
        # return ratio
        """Number of sift matches divided by 300"""
        num_matches = len(self.sift_matches) / 300.
        assert(0. <= num_matches <= 1.)
        return num_matches

    def _sift_inliers(self):
        """Ratio of inliers wrt total number of sift matches"""
        if len(self.sift_matches) == 0:
            return 0.  # give worst possible score

        src_pts = np.float32([self.kp_tg[m.queryIdx].pt for m in self.sift_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp_cs[m.trainIdx].pt for m in self.sift_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        ratio_inliers = mask.mean()
        assert(0. <= ratio_inliers <= 1.)
        return ratio_inliers

    def _sp(self):
        """SIFT Precision"""
        bbox = self.example['cs_dt_uv_bbox']

        # Size of dst_pts is [num_matches, 2]
        dst_pts = np.float32([self.kp_cs[m.trainIdx].pt for m in self.sift_matches])

        # Corner case when there are zero matches.
        # In this case, sift precision should be zero.
        if len(dst_pts) == 0:
            return 0.

        resized_cs_im_size = self.example['resized_cs_im'].size
        dst_pts_uv = dst_pts / resized_cs_im_size
        u1, v1, u2, v2 = self.example['cs_dt_uv_bbox']

        # Checking which dst_pts_uv lie inside cs_dt_uv_bbox
        dst_pts_u, dst_pts_v = dst_pts_uv[:, 0], dst_pts_uv[:, 1]
        # np.logical_and.reduce because np.logical_and only takes two arguments.
        is_dst_pt_inside_dt = np.logical_and.reduce([u1 <= dst_pts_u,
                                                     u2 >= dst_pts_u,
                                                     v1 <= dst_pts_v,
                                                     v2 >= dst_pts_v])

        sp_score = is_dst_pt_inside_dt.mean()
        assert(0. <= sp_score <= 1.)
        return sp_score

    def get_scores(self):
        # return self._sift_matches(), self._sift_inliers()
        return self._sm(), self._sp()


if __name__ == '__main__':
    import pickle
    from baselines.sift.test import test
    from flowmatch.networks.flownet import FlowNet
    from flowmatch.utils import load_config

    # Loading an (un-preprocessed) example that was saved as a pickle file.
    with open('/home/sancha/repos/osid/sandbox/example.pkl', 'rb') as f:
        example = pickle.load(f)

    cfg = load_config('../../pipeline/configs/tdid_avd2_manual_easy.yaml')
    net = FlowNet(cfg.flownet)

    sift_matches = test(net, example)
    example['pred_sift'] = sift_matches

    # Reclassification.
    reclass_scorer = SiftMatchScorer(example)
    scores = reclass_scorer.get_scores()

