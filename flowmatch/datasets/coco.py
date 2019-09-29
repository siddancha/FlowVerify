import numpy as np
import os

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision import transforms

from flowmatch.flowutils import compute_flow
from .utils import random_rotate_and_crop, id_to_str, str_to_id


class CocoDataset:
    def __init__(self, root, annFile, cfg):
        """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
        Args:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            cfg (easydict.EasyDict): Config.
        """
        self._init_transform()

        self.root = root
        self.cfg = cfg
        self.coco = COCO(annFile)
        self.img_ids = list(self.coco.imgs.keys())

        # Create a new id for every image-annotation pair.
        # A single data point is an image-annotation pair.
        print('Generating image-annotation ids ... ', end='', flush=True)

        self.ids = []
        for img_id in self.img_ids:
            img_ann = self.coco.loadImgs(img_id)[0]
            for ann_id in self.coco.getAnnIds(imgIds=img_id, iscrowd=False):
                annotation = self.coco.loadAnns(ann_id)[0]
                bbox = annotation['bbox']

                # 0 <= x1,x2 <= width, 0 <= y1,y2 <= height
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                assert (min(x1, x2) >= 0 and max(x1, x2) <= img_ann['width'])
                assert (min(y1, y2) >= 0 and max(y1, y2) <= img_ann['height'])

                # bbox_w, bbox_h = x2 - x1, y2 - y1

                self.ids.append((img_id, ann_id))
        print('done.')
        print('Total {} items in dataset.'.format(len(self.ids)))

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
        img_id, ann_id = self.ids[index]
        annotation = self.coco.loadAnns(ann_id)[0]
        fs_bbox = annotation['bbox']  # list(float) in format (x1, y1, w, h), acronym for full bbox
        # For COCO, bbox format is (x1, y1, w, h). Convert to (x1, y1, x2, y2).
        fx1, fy1, fx2, fy2 = fs_bbox[0], fs_bbox[1], fs_bbox[0] + fs_bbox[2], fs_bbox[1] + fs_bbox[3]
        fs_bbox = (fx1, fy1, fx2, fy2)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        fs_im = Image.open(os.path.join(self.root, path)).convert('RGB') # acronym for full scene

        tg_mask = self._get_target_mask(fs_im, annotation)

        # Get crop_scene and rotation matrix.
        tfs_im, tfs_bbox, cs_im, cs_mask, cs_bbox, cs_rmat = random_rotate_and_crop(
            fs_im, tg_mask, fs_bbox, self.cfg.coco.theta_range, self.cfg.coco.scale_factor_range)

        example = {'fs_im': fs_im,
                   'fs_bbox': fs_bbox,
                   'tfs_im': tfs_im,
                   'tfs_bbox': tfs_bbox,
                   'cs_im': cs_im,
                   'cs_mask': cs_mask,
                   'cs_bbox': cs_bbox,
                   'cs_rmat': cs_rmat,
                   'tg_im': fs_im,
                   'tg_mask': tg_mask,
                   'img_id': img_id,
                   'ann_id': ann_id}

        return example

    def __len__(self):
        return len(self.ids)

    def id_to_str(self, id):
        return id_to_str(id, block=8)

    def str_to_id(self, string):
        return str_to_id(id, block=8)

    def _init_transform(self):
        # Initialize torch-defined transforms that will be used by self.transform().
        self.transform_ops = {'ToTensor': transforms.ToTensor()}

    def _compute_gt_flow(self, tg_im, tg_mask, cs_rmat):
        """Computes ground truth flow for COCO.

        Args:
            tg_im (PIL.Image.Image): Target image.
            tg_mask (np.ndarray, type=np.uint8): Binary mask of target in im.
            cs_rmat (np.ndarray, np.float32, size=(3, 3)): Rotation matrix of transformation that maps (u, v) in tg_bbox
                to (u, v) in cs_bbox of crop_scene.
        """
        # Compute tg_bbox
        x_inds, y_inds = tg_mask.max(axis=0).nonzero()[0], tg_mask.max(axis=1).nonzero()[0]
        x1, x2, y1, y2 = x_inds.min(), x_inds.max(), y_inds.min(), y_inds.max()
        tg_bbox = (x1, y1, x2, y2)

        # Get flow.
        flow = compute_flow.affine_flow(tg_im.size, tg_bbox, cs_rmat, GLOBAL=True)

        # affine_flow() computed flow for the whole tg_bbox. All pixels in tg_bbox need not map to inside of cs_bbox.
        # But all pixels inside tg_mask should. Hence, flow values inside tg_mask should be between 0 and 1.
        flow *= tg_mask[:, :, None]
        tol = 0.01
        if -tol <= flow.min() and flow.max() <= 1. + tol:
            return flow
        else:
            return None  # if flow is outside tolerance limits, return None as indicator

    def add_gt_flow(self, example):
        """Takes a pre-processed example and adds GT optical flow"""

        # Get GT optical flow.
        flow = self._compute_gt_flow(example['resized_tg_im'], example['resized_tg_mask'], example['cs_rmat'])

        if flow is None:
            # If flow is not successfully computed, return None to indicate that this example has failed
            return None
            # import matplotlib.pyplot as plt
            # plt.switch_backend('agg')
            # def savefig(name='fig'):
            #     filename = '{}.png'.format(name)
            #     with open(filename, 'wb') as f: plt.savefig(f)
            #     plt.clf()
            # plt.imshow(example['fs_im']); savefig('fs_im')
            # plt.imshow(example['cs_im']); savefig('cs_im')
            # print("Flow out of tolerance bounds for image ({}, {})".format(example['img_id'], example['ann_id']))
            # print("Rotation matrix is - ")
            # print(example['cs_rmat'])

        example['flow'] = flow

        # Transform flow.

        # Step 1: Centre flow values by shifting range from [0, 1] to [-0.5, 0.5].
        centered_flow = example['flow'] - 0.5

        # Step 2: Convert array to tensor.
        example['net_flow'] = self.transform_ops['ToTensor'](centered_flow)

        return example

    def _get_target_mask(self, im, annotation):
        """
        Args:
            im: (PIL.Image) scene image.
            annotation: annotation object from calling coco.loadAnns(ann_id).
        Returns:
            tg_mask: (np.ndarray, type=np.uint8) binary mask of target in im.
        """
        seg = annotation['segmentation']  # seg mask used to crop target.
        tg_mask = np.zeros((im.height, im.width), dtype=np.bool)

        # Loop over every polygon (poly for short) in segmentation.
        # For each polygon, construct poly_mask, and or it with global mask.
        for poly in seg:  # poly is short for polygon
            try:
                pts = [int(elem) for elem in poly]
            except:
                raise ValueError('Polygon: {}'.format(poly))
            poly_im = Image.new('L', (im.width, im.height), 0)
            ImageDraw.Draw(poly_im).polygon(pts, outline=1, fill=1)
            poly_mask = np.array(poly_im, dtype=np.bool)
            tg_mask = np.logical_or(tg_mask, poly_mask)

        return tg_mask.astype(np.uint8)

# if __name__ == '__main__':
#     coco_ds = CocoDataset(root='/scratch/sancha/data/coco/images/train2014',
#                           annFile='/scratch/sancha/data/coco/annotations/instances_train2014.json')
#     dp_1 = coco_ds[0]