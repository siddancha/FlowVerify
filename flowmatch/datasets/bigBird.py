import os
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from flowmatch.datasets.utils import random_homography_and_crop, random_crop_bbox, bbox
from flowmatch.datasets.utils_syn import blend, PIL2array1C
from flowmatch.flowutils.compute_flow import affine_flow
from flowmatch.datasets.setting import *

class BigBirdDataset:
    def __init__(self, root, cfg):
        """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
        Args:
            root (string): Root directory where images are downloaded to.
            cfg (easydict.EasyDict): Config.
        """
        self.root = root
        self.cfg = cfg

        self._init_transform()

        print('Mapping all objects/masks paths ... ', end='', flush=True)
        self.data_pths = self._all_image_path()
        print('done.')
        print('Total {} items in dataset.'.format(len(self.data_pths)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Dict (image, bbox).
                bbox is the (single) bounding box annotation in xyxy format.
                bbox is a continuous float where 0 <= x1,x2 <= width and 0 <= y1,y2 <= height.

        * 'fs' is acronym for 'full_scene'.
        * 'cs' is acronym for 'crop_scene'.
        """

        target_pth, mask_pth = self.data_pths[index]
        fs_im = cv2.imread(target_pth) # acronym for full scene
        tg_mask = 255 - PIL2array1C(Image.open(mask_pth))
        x1,x2,y1,y2 = bbox(tg_mask > 0)
        h,w = x2-x1+1, y2-y1+1
        scale = min(OBJECT_SIZE / max(h,w), 1)
        tg_im = cv2.resize(fs_im[x1:x2+1, y1:y2+1], (int(w*scale), int(h*scale)))
        tg_mask = cv2.resize(tg_mask[x1:x2+1, y1:y2+1], (int(w*scale), int(h*scale)))

        # Get crop_scene and homography matrix.
        cs_im, cs_mask, M = random_homography_and_crop(tg_im, tg_mask)
        x1,x2, y1,y2 = bbox(cs_mask > 0)
        dx, dy = y1, x1

        # Get homography flow
        h,w = cs_im.shape[:2]
        flow = affine_flow([w,h], [0,0,w-1,h-1], M)

        # Paste object into background
        cs_info = blend(Image.fromarray(cs_im), Image.fromarray(cs_mask))
        bbx, cs_im, cs_mask = cs_info['bbx'], cs_info['cs_im'], cs_info['cs_mask']

        # Crop object randomly from the blended image
        x1, x2, y1, y2 = bbx
        cs_mask = np.array(cs_mask)
        hp, wp = cs_im.height, cs_im.width
        cs_im, cs_mask, cs_bbox = random_crop_bbox(cs_im, cs_mask, [x1, y1, x2, y2], [1.2, 1.6])
        
        # Translate flow accordingly
        xp1, yp1, xp2, yp2 = np.multiply(cs_bbox, np.array([wp, hp, wp, hp]))
        hp, wp = cs_im.height, cs_im.width
        tg_mask = np.float32(tg_mask > 0)
        flow[:,:,0] = (flow[:,:,0]*w + x1 - xp1 - dx) / wp
        flow[:,:,1] = (flow[:,:,1]*h + y1 - yp1 - dy) / hp

        example = {'cs_im': cs_im,
                   'cs_mask': cs_mask,
                   'homography': M,
                   'tg_im': tg_im,
                   'tg_mask': tg_mask,
                   'flow': flow,
                   'img_id': target_pth,
                   }
        return example

    def __len__(self):
        return len(self.data_pths)

    def _init_transform(self):
        # Initialize torch-defined transforms that will be used by self.transform().
        self.transform_ops = {'ToTensor': transforms.ToTensor(),
                              'Normalize': transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)}

    def _all_image_path(self):
        objects = list(filter(lambda x: os.path.isdir(os.path.join(self.root, x)), 
            os.listdir(self.root)))
        res = []
        for obj in objects:
            obj_root = os.path.join(self.root, obj)
            img_pths = list(filter(lambda x: x.endswith('.jpg'), os.listdir(obj_root)))
            data = list(map(lambda x: (os.path.join(obj_root, x), 
                                        os.path.join(obj_root, 'masks', x[:-4]+'_mask.pbm')),
                        img_pths))
            data = list(filter(lambda x: os.path.exists(x[0]) and os.path.exists(x[1]), data))
            # print('Object {} with {} images'.format(obj, len(data)))
            res += data
        return res

    def add_gt_flow(self, example):
        # Get GT optical flow.
        # flow = self._compute_gt_flow(example['resized_tg_im'], example['resized_tg_mask'], example['homography'])

        if 'flow' not in example.keys():
            print('flow unavailable for {}'.format(example['img_id']))
            # If flow is not successfully computed, return None to indicate that this example has failed
            return None

        # Transform flow.

        # Step 1: Centre flow values by shifting range from [0, 1] to [-0.5, 0.5].
        centered_flow = example['resized_flow'] - 0.5

        # Step 2: Convert array to tensor.
        example['net_flow'] = self.transform_ops['ToTensor'](centered_flow)
        return example
