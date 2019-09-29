import argparse
import numpy as np
from tqdm import tqdm

from flowmatch.datasets.coco  import CocoDataset
from flowmatch.networks.flownet import FlowNet
from flowmatch.flowutils.utils import get_identity_flow
from flowmatch.utils import load_config
from flowmatch.exps.coco.main import coco_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing baseline performance on COCO')
    parser.add_argument('--config', default='config', help='name of config')
    parser.add_argument('--num_examples', '-r', type=int, default=None, help='number of examples to test on')
    args = parser.parse_args()

    cfg_path = '../../exps/coco/{}.py'.format(args.config)
    cfg = load_config(cfg_path)

    coco_valid = CocoDataset(root=cfg.coco.valid.image_dir, annFile=cfg.coco.valid.ann_file, cfg=cfg)
    print('Filtering training set ... ', end='', flush=True)
    filtered_ids = coco_filter(coco_valid)
    coco_valid.item_ids = filtered_ids
    print('done.')

    net = FlowNet(cfg)  # just for pre-processing data

    # Variables for identity flow
    sum_identity_epe = 0.0
    identity_flow = get_identity_flow(cfg.tg_size, cfg.tg_size)

    # Variables for best constant flow
    sum_gt_flow, sum_sq_gt_flow, sum_tg_mask = 0., 0., 0.

    indices = iter(np.random.permutation(len(coco_valid)))
    for i in tqdm(range(args.num_examples)):
        example = None
        while example is None:
            index = next(indices)
            example = coco_valid[index];
            example = net.preprocess(example)
            example = coco_valid.add_gt_flow(example)

        gt_flow, tg_mask = example['flow'], example['resized_tg_mask']
        tg_mask = tg_mask.reshape(cfg.tg_size, cfg.tg_size, 1)

        # Compute EPE loss for identity flow.
        identity_diff = tg_mask * (gt_flow - identity_flow)
        identity_norm = np.linalg.norm(identity_diff, ord=2, axis=2)
        identity_epe = identity_norm.sum() / tg_mask.sum()
        sum_identity_epe += identity_epe

        # Compute running sums for best constant flow.
        sum_gt_flow = sum_gt_flow + tg_mask * gt_flow
        sum_sq_gt_flow = sum_sq_gt_flow + tg_mask * (gt_flow ** 2)
        sum_tg_mask = sum_tg_mask + tg_mask

    identity_epe = sum_identity_epe / args.num_examples

    mean_gt_flow = sum_gt_flow / args.num_examples
    mean_sq_gt_flow = sum_sq_gt_flow / args.num_examples
    mean_constant_epe =  mean_sq_gt_flow - (mean_gt_flow ** 2)
    sum_constant_epe = tg_mask * mean_constant_epe
    constant_epe = np.sqrt(sum_constant_epe.sum() / tg_mask.sum())

    print("EPE for identity flow:                  [[{:.3f}]].".format(identity_epe))
    print("Upper bound EPE for best constant flow: [[{:.3f}]].".format(constant_epe))


