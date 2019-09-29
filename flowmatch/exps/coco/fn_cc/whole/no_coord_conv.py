import os
from easydict import EasyDict as edict

# Exp path settings
exp_root = '/scratch/sancha/exp/FlowMatch'
data_root = '/scratch/sancha/data'

# Seed
seed = 1

# COCO settings
coco = edict()
coco.root = os.path.join(data_root, 'coco')
coco.train, coco.valid = edict(), edict()
coco.train.image_dir = os.path.join(coco.root, 'images/train2014')
coco.train.ann_file = os.path.join(coco.root, 'annotations/instances_train2014.json')
coco.valid.image_dir = os.path.join(coco.root, 'images/val2014')
coco.valid.ann_file = os.path.join(coco.root, 'annotations/instances_val2014.json')

# COCO pre-processing settings
coco.theta_range = (-45., 45.)
coco.scale_factor_range = (1.2, 1.6)

# Architecture settings
arch = 'FlowNetC'
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
coord_conv = False

# Size settings
cs_size = 256
tg_size = 256

# Train settings
batch_size = 2 # original - 8
optimizer = 'sgd'
lr = 3e-4 # starting LR
weight_decay = 1e-4
num_iter = 800000
lr_changes = {385000: 1e-4}
ckpt_every = 5000
num_valid = 500


# Validation settings
eval_while_training = False