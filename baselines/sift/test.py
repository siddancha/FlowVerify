import cv2
import numpy as np


def test(net, example):
    """
    Args:
        net (FlowNet): Instance of networks.flownet.FlowNet model, only to be used for pre-processing.
        example (dict): Un-processed example.
    Returns:
        good (list, DMatch): List of good SIFT matches.
    """
    net.eval()

    example = net.preprocess(example)
    cs_arr, tg_arr = np.array(example['resized_cs_im']), np.array(example['resized_tg_im'])
    cs_mask, tg_mask = example['resized_cs_mask'], example['resized_tg_mask']

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp_cs, des_cs = sift.detectAndCompute(cs_arr, mask=cs_mask)
    kp_tg, des_tg = sift.detectAndCompute(tg_arr, mask=tg_mask)

    example['kp_cs'], example['kp_tg'] = kp_cs, kp_tg

    # Return empty list no matches if no matches are found in either scene or target.
    if des_cs is None or des_tg is None:
        return []

    # Make sure that there are at-least 2 features in both scene and target for knn with nn=2.
    if len(des_cs) < 2 or len(des_tg) < 2:
        return []

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_tg, des_cs, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good


if __name__ == '__main__':
    import pickle
    from flowmatch.networks.flownet import FlowNet
    from flowmatch.utils import load_config

    # Loading an (un-preprocessed) example that was saved as a pickle file.
    with open('/home/sancha/repos/osid/sandbox/example.pkl', 'rb') as f:
        example = pickle.load(f)

    cfg = load_config('../../pipeline/configs/tdid_avd2_manual_easy.yaml')
    net = FlowNet(cfg.flownet)

    good = test(net, example)