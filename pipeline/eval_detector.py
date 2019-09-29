import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


class MapEvaluator:
    def __init__(self, gt_json, cat_ids):
        # Eval options
        self.iou_threshs = [.5]
        self.max_dets = [1, 10, 100]

        self.cat_ids = cat_ids

        self.coco_gt = COCO(gt_json)

    def _init_coco_eval(self, dt_json_data):
        coco_dt = self.coco_gt.loadRes(dt_json_data)

        self.coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        self.coco_eval.params.iouThrs = np.array(self.iou_threshs)
        self.coco_eval.params.maxDets = self.max_dets
        self.coco_eval.params.catIds = self.cat_ids

    def do_eval(self, dt_json_data):
        self._init_coco_eval(dt_json_data)

        self.coco_eval.evaluate()
        self.coco_eval.accumulate()

    def do_eval_and_print(self, dt_json_data):
        self._init_coco_eval(dt_json_data)

        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

    def get_avg_precision_recall(self, t=0, a=0, m=-1):
        p = self.coco_eval.eval['precision'][t, :, :, a, m]
        r = self.coco_eval.eval['recall'][t, :, a, m]

        # Removing -1 entries for classes that have no GT objects.
        p = p[:, np.where(r>-1)[0]]
        r = r[r>-1]
        assert(np.all(p >= 0.) and np.all(r >= 0.))

        ap = p.mean(axis=1)
        ar = r.mean()

        return ap, ar

    def evaluateImg(self, imgId, catId, aRng=(-np.inf, np.inf), maxDet=100):
        p = self.coco_eval.params
        # add backward compatibility if useSegm is specified in params
        p.iouType = 'bbox'
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))

        p.maxDets = sorted(p.maxDets)
        self.coco_eval.params=p

        self.coco_eval._prepare()

        self.coco_eval.ious = {(imgId, catId): self.coco_eval.computeIoU(imgId, catId)}
        img_eval = self.coco_eval.evaluateImg(imgId, catId, aRng, maxDet)
        inds = img_eval['dtIds']
        matches = img_eval['dtMatches']

        return inds, matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to compute detection accuracy')
    parser.add_argument('--gt-json', help='GT json file path')
    parser.add_argument('--dt-json', help='Detector output json file path')
    parser.set_defaults()
    args = parser.parse_args()

    gmu_ids = [5, 10, 12, 14, 18, 21, 28, 50, 79, 94, 96]

    eval = MapEvaluator(args.gt_json, gmu_ids)
    eval.do_eval_and_print(args.dt_json)
    ap, ar = eval.get_avg_precision_recall()
    # np.set_printoptions(threshold=sys.maxsize)
    # print(ap)
    recthrs = eval.coco_eval.params.recThrs