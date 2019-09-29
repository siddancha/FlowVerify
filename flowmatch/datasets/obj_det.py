import argparse
import json
import numpy as np
import os
import time
from tqdm import tqdm

from PIL import Image
from torchvision import transforms
from flowmatch.datasets.utils import id_to_str, str_to_id, center_crop, xywh2xyxy
from flowmatch.networks.flownet import FlowNet
from flowmatch.utils import load_config


class TargetStore:
    def __init__(self, obj_json, obj_root, cfg):
        with open(obj_json, 'r') as f:
            self.obj_data = json.load(f)
        self.obj_root = obj_root
        self.cache_root = cfg.obj_det.obj_cache
        self.cfg = cfg

        self.net = FlowNet(self.cfg.flownet)  # for accessing pre-processing functions
        self.store = None

    def __getitem__(self, category_id):
        return self.store[category_id]

    def _create_example_from_scratch(self, tg_im_file, tg_mask_file, vp_name):
        tg_im_path = os.path.join(self.obj_root, tg_im_file)
        tg_mask_path = os.path.join(self.obj_root,tg_mask_file)

        tg_im = Image.open(tg_im_path)

        tg_mask = Image.open(tg_mask_path)
        tg_mask = np.array(tg_mask).astype(np.uint8)
        tg_mask = (tg_mask / tg_mask.max()).astype(tg_mask.dtype)
        if self.cfg.obj_det.mask_inverted: tg_mask = 1 - tg_mask

        example = {'tg_im': tg_im, 'tg_mask': tg_mask, 'vp_name': vp_name}
        example = self.net.preprocess_target(example, do_step1=True)
        return example

    def _create_example_from_cache(self, tg_im_file, tg_mask_file, vp_name):
        resized_tg_im_path = os.path.join(self.cache_root, tg_im_file)
        resized_tg_mask_path = os.path.join(self.cache_root, tg_mask_file)

        resized_tg_im = Image.open(resized_tg_im_path)
        resized_tg_mask = np.array(Image.open(resized_tg_mask_path))

        # Contents of example are that post step1.
        example = {'resized_tg_im': resized_tg_im, 'resized_tg_mask': resized_tg_mask, 'vp_name': vp_name}
        example = self.net.preprocess_target(example, do_step1=False)

        return example

    def _save_example_to_cache(self, tg_im_file, tg_mask_file, example):
        resized_tg_im_path = os.path.join(self.cache_root, tg_im_file)
        resized_tg_mask_path = os.path.join(self.cache_root, tg_mask_file)

        resized_tg_im, resized_tg_mask = example['resized_tg_im'], example['resized_tg_mask']

        def create_dir(path):
            dir = os.path.dirname(path)
            os.makedirs(dir, exist_ok=True)

        create_dir(resized_tg_im_path)
        resized_tg_im.save(resized_tg_im_path)

        create_dir(resized_tg_mask_path)
        Image.fromarray(resized_tg_mask).save(resized_tg_mask_path)

    def create_tg_store(self, use_cache):
        """Store contains pre-loaded and pre-processed target images and masks"""
        start = time.time()

        tg_store = {}  # key: category-id, value: list of target example dictionaries.
        for obj_elem in self.obj_data:
            category_id = obj_elem['category_id']
            list_example = []
            print('Loading and pre-processing object {} ...'.format(obj_elem['category_name']))
            for vp_id in tqdm(range(len(obj_elem['image_files']))):
                tg_im_file, tg_mask_file, vp_name =\
                    obj_elem['image_files'][vp_id], obj_elem['mask_files'][vp_id], obj_elem['vp_names'][vp_id]
                if use_cache:
                    example = self._create_example_from_cache(tg_im_file, tg_mask_file, vp_name)
                else:
                    example = self._create_example_from_scratch(tg_im_file, tg_mask_file, vp_name)

                list_example.append(example)

            tg_store[category_id] = list_example

        lp_time = time.time() - start
        print('Total loading and processing time: {:.1f}s'.format(lp_time))
        self.store = tg_store

    def create_cache(self):
        start = time.time()

        for obj_elem in self.obj_data:
            print('Loading and pre-processing object {} ...'.format(obj_elem['category_name']))
            for vp_id in tqdm(range(len(obj_elem['image_files']))):
                tg_im_file, tg_mask_file, vp_name =\
                    obj_elem['image_files'][vp_id], obj_elem['mask_files'][vp_id], obj_elem['vp_names'][vp_id]
                example = self._create_example_from_scratch(tg_im_file, tg_mask_file, vp_name)
                self._save_example_to_cache(tg_im_file, tg_mask_file, example)

        lp_time = time.time() - start
        print('Total loading and processing time: {:.1f}s'.format(lp_time))
        print('Cache created.')


class ObjectDetectionDataset:
    def __init__(self, gt_json, dt_json, obj_json, img_root, obj_root, cfg, use_cache=True):
        """Dataset based on detections of an object detector and images of objects
        Args:
            gt_json (string): Path to json file of GT annotations, in COCO format.
            dt_json (string): Path to json file of detections, in COCO format.
            obj_json (string): Path to json file that describes objects.
            img_root (string): Path where scene images are located.
            obj_root (string): Path where object images are located.
            cfg (easydict.EasyDict): Config.
        """
        self._init_transform()

        with open(gt_json, 'r') as f:
            self.gt_data = json.load(f)
        with open(dt_json, 'r') as f:
            self.dt_data = json.load(f)
        self.img_root = img_root
        self.cfg = cfg

        self.tg_store = TargetStore(obj_json, obj_root, cfg)
        self.tg_store.create_tg_store(use_cache)

        self.ids, self.dt_ids = [], []
        for dt_elem in self.dt_data:
            dt_id = dt_elem['id']
            self.dt_ids.append(dt_id)

            category_id = dt_elem['category_id']
            obj_elem = self._obj_elem_from_category_id(category_id)
            num_vp = len(obj_elem['image_files'])
            for vp_id in range(num_vp):
                self.ids.append((dt_id, vp_id))

        # Cache scene image to be reused instead of reloading.
        self.cached_fs = (None, None)  # (path, PIL.Image.Image)

    def _obj_elem_from_category_id(self, category_id):
        obj_category_ids = [e['category_id'] for e in self.tg_store.obj_data]
        index = obj_category_ids.index(category_id)
        return self.tg_store.obj_data[index]

    def _load_fs_im(self, path):
        cached_path, cached_im = self.cached_fs
        if cached_path != path:
            print('Loading scene image ...')
            self.cached_fs = path, Image.open(path)
        return self.cached_fs[1]

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
        return self.get_example_from_id(self.ids[index])

    def get_scene_example_from_dt_id(self, dt_id):
        dt_index = [e['id'] for e in self.dt_data].index(dt_id)
        dt_elem = self.dt_data[dt_index]
        category_id = dt_elem['category_id']
        correct = dt_elem['correct'] if 'correct' in dt_elem else False
        fs_gt_xy_bbox = xywh2xyxy(dt_elem['gt_bbox']) if ('gt_bbox' in dt_elem and dt_elem['gt_bbox'] is not None) else None
        gt_vp_name = dt_elem['gt_vp_name'] if 'gt_vp_name' in dt_elem else None

        gt_image_elem = [e for e in self.gt_data['images'] if e['id'] == dt_elem['image_id']][0]
        img_path = os.path.join(self.img_root, gt_image_elem['file_name'])
        fs_im = self._load_fs_im(img_path)

        # Cropping square bbox from full scene.
        fs_dt_xy_bbox = xywh2xyxy(dt_elem['bbox'])
        cs_im, fs_cs_xy_bbox = center_crop(fs_im, fs_dt_xy_bbox)

        # Compute cs_dt_uv_bbox.
        cs_w, cs_h = cs_im.size
        fs_dt_x1, fs_dt_y1, fs_dt_x2, fs_dt_y2 = fs_dt_xy_bbox
        fs_cs_x1, fs_cs_y1, fs_cs_x2, fs_cs_y2 = fs_cs_xy_bbox
        cs_dt_x1, cs_dt_y1, cs_dt_x2, cs_dt_y2 = \
            fs_dt_x1 - fs_cs_x1, fs_dt_y1 - fs_cs_y1, fs_dt_x2 - fs_cs_x1, fs_dt_y2 - fs_cs_y1
        cs_dt_u1, cs_dt_v1, cs_dt_u2, cs_dt_v2 = cs_dt_x1 / cs_w, cs_dt_y1 / cs_h, cs_dt_x2 / cs_w, cs_dt_y2 / cs_h
        cs_dt_uv_bbox = (cs_dt_u1, cs_dt_v1, cs_dt_u2, cs_dt_v2)

        # TODO: mask should be 0 in region that was padded to make it a square.
        cs_mask = np.ones(cs_im.size, dtype=np.uint8) # TODO: actually make non-ones mask?

        return {'fs_im': fs_im,
                'fs_cs_xy_bbox': fs_cs_xy_bbox,
                'fs_dt_xy_bbox': fs_dt_xy_bbox,
                'cs_dt_uv_bbox': cs_dt_uv_bbox,
                'cs_im': cs_im,
                'cs_mask': cs_mask,
                'dt_id': dt_id,
                'category_id': category_id,
                'correct': correct,
                'fs_gt_xy_bbox': fs_gt_xy_bbox,
                'gt_vp_name': gt_vp_name}

    def get_example_from_id(self, id):
        dt_id, vp_id = id

        scene_example = self.get_scene_example_from_dt_id(dt_id)
        category_id = scene_example['category_id']

        target_example = self.tg_store[category_id][vp_id]

        example = {**scene_example, **target_example}  # merge both dictionaries
        return example

    def __len__(self):
        return len(self.ids)

    def id_to_str(self, id):
        return id_to_str(id, block=8)

    def str_to_id(self, string):
        return str_to_id(string, block=8)

    def _init_transform(self):
        # Initialize torch-defined transforms that will be used by self.transform().
        self.transform_ops = {'ToTensor': transforms.ToTensor()}

    def add_gt_flow(self, example):
        return example


if __name__ == '__main__':
    """For creating cache"""
    parser = argparse.ArgumentParser(description='Creating an object dataset cache.')
    parser.add_argument('--config', default='config', help='config file path')
    args = parser.parse_args()

    cfg = load_config(args.config)

    tg_store = TargetStore(obj_json=cfg.obj_det.obj_json, obj_root=cfg.obj_det.obj_root, cfg=cfg)
    tg_store.create_cache()
