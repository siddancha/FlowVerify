'''
Author: Siddharth Ancha
'''

import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms

from PIL import  Image
from flowmatch.utils import crop_and_resize_target


class FlowNet(nn.Module):
    def __init__(self, cfg):
        super(FlowNet, self).__init__()
        self.cfg = cfg
        self._init_transform()

    def _init_transform(self):
        # Initialize torch-defined transforms that will be used by self.transform().
        self.transform_ops = {}
        self.transform_ops['ToTensor'] = transforms.ToTensor()
        self.transform_ops['Normalize'] = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                               std=(0.229, 0.224, 0.225))

    def preprocess_scene(self, example):
        """Pre-processes scene example for preparing training item.

        Adds entries to example dict. Entries with prefix "net" can be input to neural network.
        """

        # STEP 1: Resize crop_scene.
        # --------------------------

        # Compute scale for resizing.
        cs_im, cs_mask = example['cs_im'], example['cs_mask']
        assert (cs_im.size[0] == cs_im.size[1])  # crop_scene should be square
        assert(cs_im.size == cs_mask.shape)  # size of image and mask should be the same
        cs_scale = float(self.cfg.cs_size) / float(cs_im.size[0])
        example['cs_scale'] = cs_scale

        # Resize crop scene image.
        resized_cs_im = cs_im.resize((self.cfg.cs_size, self.cfg.cs_size), Image.BILINEAR)
        example['resized_cs_im'] = resized_cs_im

        # Resized crop scene mask.
        resized_cs_mask = Image.fromarray(cs_mask)
        resized_cs_mask = resized_cs_mask.resize((self.cfg.cs_size, self.cfg.cs_size), Image.NEAREST)
        resized_cs_mask = np.array(resized_cs_mask)
        example['resized_cs_mask'] = resized_cs_mask

        # There is no need to resize example['crop_bbox'] as it is represented as fractions of crop_scene dims.

        # STEP 2: Convert arrays to tensors.
        # ----------------------------------

        example['tensor_cs_im'] = self.transform_ops['ToTensor'](example['resized_cs_im'])


        # STEP 3: Normalize RGB images.
        # -----------------------------

        # Normalize cropped scene image and zero background.
        # Zeroing after normalization corresponds to mean intensity in background.
        net_cs_im = self.transform_ops['Normalize'](example['tensor_cs_im'])
        y_inds, x_inds = (example['resized_cs_mask'] == 0).nonzero()  # indices where mask is 0
        net_cs_im[:, y_inds, x_inds] = 0  # tensor dimensions are 3xHxW
        example['net_cs_im'] = net_cs_im

        example['scene_is_processed'] = None  # add processed flag
        return example

    def preprocess_target(self, example, do_step1=True):
        """Pre-processes target example for preparing training item.

        Adds entries to example dict. Entries with prefix "net" can be input to neural network.
        """

        # STEP 1: Crop and resize target image, mask and other arrays.
        # --------------------------

        if do_step1:
            if 'flow' in example:
                tg_mask, tg_arr, flow = example['tg_mask'], np.array(example['tg_im']), example['flow']
                tg_mask, [tg_arr, flow] = crop_and_resize_target(tg_mask, [tg_arr, flow], self.cfg.tg_size)
                example['resized_tg_mask'], example['resized_tg_im'], example['resized_flow'] = tg_mask, Image.fromarray(tg_arr.astype('uint8')), flow
            else:
                tg_mask, tg_arr = example['tg_mask'], np.array(example['tg_im'])
                tg_mask, [tg_arr] = crop_and_resize_target(tg_mask, [tg_arr], self.cfg.tg_size)
                example['resized_tg_mask'], example['resized_tg_im'] = tg_mask, Image.fromarray(tg_arr)

        # STEP 2: Convert arrays to tensors.
        # ----------------------------------

        example['tensor_tg_im'] = self.transform_ops['ToTensor'](example['resized_tg_im'])
        example['net_tg_mask'] = torch.from_numpy(example['resized_tg_mask']).float()

        # STEP 3: Normalize RGB images.
        # -----------------------------

        # Normalize target image and zero background.
        # Zeroing after normalization corresponds to mean intensity in background.
        net_tg_im = self.transform_ops['Normalize'](example['tensor_tg_im'])
        y_inds, x_inds = (example['resized_tg_mask'] == 0).nonzero()  # indices where mask is 0
        net_tg_im[:, y_inds, x_inds] = 0  # tensor dimensions are 3xHxW
        example['net_tg_im'] = net_tg_im

        example['target_is_processed'] = None  # add processed flag
        return example

    def preprocess(self, example):
        """Pre-processes example for preparing training item.

        Adds entries to example dict. Entries with prefix "net" can be input to neural network.
        """
        if 'scene_is_processed' not in example:
            example = self.preprocess_scene(example)

        if 'target_is_processed' not in example:
            example = self.preprocess_target(example)

        return example


    @staticmethod
    def collate_fn(list_example):
        """Collates individual inputs and outputs into a batch
        Args:
            list_example (list(dict)): List of examples, where each example is a data point.
        Returns:
            cs_input (Tensor, type=float32, shape=Bx3xNxN): Cropped scene image.
            tg_input (Tensor, type=float32, shape=Bx3xNxN): Target image.
            tg_mask (Tensor, type=float32, shape=BxNxN): Target image mask.
            flow_target (Tensor, type=float32, shape=BxNxN): Ground truth flow.
        """
        cs_input = torch.stack([example['net_cs_im'] for example in list_example])
        tg_input = torch.stack([example['net_tg_im'] for example in list_example])
        tg_mask = torch.stack([example['net_tg_mask'] for example in list_example])
        flow_target = torch.stack([example['net_flow'] for example in list_example])
        img_pths = ([example['img_id'] for example in list_example])

        return cs_input, tg_input, tg_mask, flow_target, img_pths
