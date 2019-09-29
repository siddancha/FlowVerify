import os
from easydict import EasyDict as edict

# Exp path settings
exp_root = '/scratch/jnan1/exp/FlowMatch'
data_root = '/scratch/jnan1/'

# Seed
seed = 1

# FlyingThings3D settings
bigBird = edict()
bigBird.root = os.path.join(data_root, 'BIGBIRD')
bigBird.train, bigBird.valid = edict(), edict()
bigBird.train.image_dir = os.path.join(bigBird.root, 'TRAIN')
bigBird.valid.image_dir = os.path.join(bigBird.root, 'VALID')

# COCO pre-processing settings
# coco.theta_range = (-45., 45.)
# coco.scale_factor_range = (1.2, 1.6)

# Architecture settings
arch = 'FlowNetC'
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
coord_conv = True

# Size settings
cs_size = 256
tg_size = 256

# Train settings
batch_size = 2 # original - 8
optimizer = 'adam'
lr = 1e-6 # starting LR
weight_decay = 0
num_iter = 3200000
lr_changes = {}
ckpt_every = 500
num_valid = 500


# Validation settings
eval_while_training = False