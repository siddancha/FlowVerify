import argparse
import importlib.util
import os
from tensorboardX import SummaryWriter

from flowmatch.datasets.bigBird import BigBirdDataset
from flowmatch.train import Trainer
from flowmatch.validator import Validator
from flowmatch.datastream import SingleDataStream

from flowmatch.networks.flownet_simple import FlowNetS
from flowmatch.networks.flownet_cc import FlowNetC

def main():
	# ############# MAIN #############
	parser = argparse.ArgumentParser(description='Training FlowNetS on BIGBIRD dataset')
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
	bigBird_train = BigBirdDataset(root=cfg.bigBird.train.image_dir, cfg=cfg)

	# Validation set.
	bigBird_valid = BigBirdDataset(root=cfg.bigBird.valid.image_dir, cfg=cfg)

	# Experiment directory.
	exp_dir = os.path.join(cfg.exp_root, 'bigBird', args.config)

	# Configure logger.
	tb_dir = os.path.join(exp_dir, 'tb')
	train_writer = SummaryWriter(os.path.join(tb_dir, 'train'))

	# Create Validator
	# validator = Validator(cfg, tb_dir)
	validator = Validator([bigBird_train, bigBird_valid], ['train_epe', 'valid_epe'], cfg, tb_dir)


	def get_example_fn(index):
	    example = bigBird_train[index]
	    if example is None: return example
	    example = net.preprocess(example)
	    example = bigBird_train.add_gt_flow(example)
	    return example


	trainstream = SingleDataStream(bigBird_train, get_example_fn=get_example_fn, collate_fn=net.collate_fn)

	# Train.
	trainer = Trainer(exp_dir, cfg, net, trainstream, train_writer, validator, args.resume)
	trainer.train()

if __name__ == '__main__':
	main()


