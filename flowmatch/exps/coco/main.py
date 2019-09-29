import argparse
import importlib.util
import os
from tensorboardX import SummaryWriter

from flowmatch.datasets import CocoDataset, coco_filter
from flowmatch.train import Trainer
from flowmatch.datastream import SingleDataStream
from flowmatch.validator import Validator

from flowmatch.networks.flownet_simple import FlowNetS
from flowmatch.networks.flownet_cc import FlowNetC

# ############# MAIN #############
def main():
    parser = argparse.ArgumentParser(description='Training FlowNetS on COCO dataset')
    parser.add_argument('--config', default='config', help='name of config')
    parser.add_argument('--resume', '-r', type=int, default=None, help='resume from checkpoint')
    args = parser.parse_args()

    # Loading config.
    cfg_path = '{}.py'.format(args.config)
    spec = importlib.util.spec_from_file_location('cfg', cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    print('Building model...')
    if cfg.arch == 'FlowNetS':
        net = FlowNetS(input_channels=6, batchNorm=False, cfg=cfg).cuda()
    elif cfg.arch == 'FlowNetC':
        net = FlowNetC(batchNorm=False, cfg=cfg).cuda()
    else:
        raise Exception("cfg.arch must be one of [FlowNetC, FlowNetS]")

    # Training set.
    coco_train = CocoDataset(root=cfg.coco.train.image_dir, annFile=cfg.coco.train.ann_file, cfg=cfg)
    print('Filtering training set ... ', end='', flush=True)
    filtered_ids = coco_filter(coco_train)
    coco_train.ids = filtered_ids
    print('done.')

    # Validation set.
    coco_valid = CocoDataset(root=cfg.coco.valid.image_dir, annFile=cfg.coco.valid.ann_file, cfg=cfg)
    print('Filtering validation set ... ', end='', flush=True)
    filtered_ids = coco_filter(coco_valid)
    coco_valid.ids = filtered_ids
    print('done.')

    # # A selection of three data points.
    # coco_train.ids = [(69675, 675057), (90570, 1537674), (311914, 271127)]

    # Experiment directory.
    exp_dir = os.path.join(cfg.exp_root, 'coco', args.config)

    # Configure logger.
    tb_dir = os.path.join(exp_dir, 'tb')
    train_writer = SummaryWriter(os.path.join(tb_dir, 'train'))

    # Create Validator
    validator = Validator([coco_train, coco_valid], ['train_epe', 'valid_epe'], cfg, tb_dir)


    def get_example_fn(index):
        example = coco_train[index]
        example = net.preprocess(example)
        example = coco_train.add_gt_flow(example)
        return example


    trainstream = SingleDataStream(coco_train, get_example_fn=get_example_fn, collate_fn=net.collate_fn)

    # Train.
    trainer = Trainer(exp_dir, cfg, net, trainstream, train_writer, validator, args.resume)
    trainer.train()
    
if __name__ == '__main__':
    main()
