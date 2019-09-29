from .bigBird import BigBirdDataset
from .coco import CocoDataset
from .obj_det import ObjectDetectionDataset

def coco_filter(ds):
    """
    Filter all (im, bbox) pairs where segmentation area is greater than bounding
    box area, and where larger side of bbox is greater than smaller side of image.
    """
    num_bad_area, num_large_bbox_side = 0, 0
    filtered_ids = []

    for img_id, ann_id in ds.ids:
        img_ann = ds.coco.loadImgs(img_id)[0]
        min_im_side = min(img_ann['width'], img_ann['height'])

        annotation = ds.coco.loadAnns(ann_id)[0]
        seg_area, bbox = annotation['area'], annotation['bbox']

        # Incorrect annotation, segmentation area is greater than
        # bounding box area. Do not add these examples to dataset.
        if seg_area > bbox[2] * bbox[3]:
            num_bad_area += 1
            continue

        # Check if larger bbox side is larger than smaller img side.
        bbox_w, bbox_h = bbox[2], bbox[3]
        max_bbox_side = max(bbox_w, bbox_h)
        if max_bbox_side > min_im_side - 1:
            num_large_bbox_side += 1
            continue

        filtered_ids.append((img_id, ann_id))

    # Printing stats.
    total_pairs = len(ds.ids)
    print('Total {} image-bbox pairs.'.format(total_pairs))
    print('Removed {} ({:.2f}%) image-bbox pairs with incorrect area annotation.'.format(
        num_bad_area, (num_bad_area / total_pairs) * 100.))
    print('Removed {} ({:.2f}%) image-bbox pairs with large bbox side.'.format(
        num_large_bbox_side, (num_large_bbox_side / total_pairs) * 100.))
    print('Size of filtered dataset (image-bbox pairs) is {}.'.format(len(filtered_ids)))

    return filtered_ids

def load_dataset(ds_name, cfg):
    """Load one of different kinds of datasets"""
    if ds_name == 'bigBird':
        dataset = BigBirdDataset(root=cfg.bigBird.valid.image_dir, cfg=cfg)
    elif ds_name == 'coco':
        dataset = CocoDataset(root=cfg.coco.valid.image_dir, annFile=cfg.coco.valid.ann_file, cfg=cfg)
        print('Filtering coco validation set ... ', end='', flush=True)
        filtered_ids = coco_filter(dataset)
        dataset.item_ids = filtered_ids
        print('done.')
    elif ds_name == 'object_detection':
        dataset = ObjectDetectionDataset(gt_json=cfg.obj_det.gt_json, dt_json=cfg.obj_det.dt_json,
                                         obj_json=cfg.obj_det.obj_json, img_root=cfg.obj_det.img_root,
                                         obj_root=cfg.obj_det.obj_root, cfg=cfg)
    else:
        raise Exception('--dataset must be one of [bigBird, coco, flyingThings3D, object_detection]')
    return dataset
