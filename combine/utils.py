import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt

from time import time
import argparse
import os

import numpy as np
import json
import copy
from contextlib import redirect_stdout
from pipeline.eval_detector import MapEvaluator
from contextlib import redirect_stdout

def rerank_dets(dt_json_data, score_json_data):
    # Dictionarize score_json
    score_json_dict = {entry['id']: entry for entry in score_json_data}

    merged_json_data = []
    for dt_json_entry in dt_json_data:
        dt_id, dt_score = dt_json_entry['id'], dt_json_entry['score']
        score_json_score = max(score_json_dict[dt_id]['scores']) # max scores across viewpoints.

        merged_json_entry = dt_json_entry.copy()
        merged_json_entry['passed'] = score_json_score > 0 
        merged_json_data.append(merged_json_entry)

    merged_json_data.sort(key=lambda x: (x['passed'], x['score']), reverse=True)

    count = len(merged_json_data)
    for i in range(len(merged_json_data)-1):
        if merged_json_data[i]['score'] != merged_json_data[i+1]['score']:
            count -= 1
        merged_json_data[i]['score'] = count

    return merged_json_data

def _do_mAP_eval(gt_json_file, dt_json_data, is_gmu=False, cat_ids=None):
    eva = MapEvaluator(gt_json_file, cat_ids)
    eva.do_eval(dt_json_data)
    ap, ar = eva.get_avg_precision_recall()
    params = eva.coco_eval.params
    recs = params.recThrs
    return ap, ar, recs

def evaluate(gt_json, data, is_gmu, cat_ids=None):
    with redirect_stdout(None):
        ap, ar, recs = _do_mAP_eval(gt_json, copy.deepcopy(data), is_gmu=is_gmu, cat_ids=cat_ids)
    return ap, ar, recs
  
def plot_pr_curves(aps, names, recs, out_dir, 
                    file_name, colors, styles, threshold=0.7):
    plt.figure()
    plt.xlabel('Recall', size='xx-large')
    plt.ylabel('Precision', size='xx-large')
    title = file_name.upper().replace('EASY', 'Large')

    # plt.title(title, size='xx-large')
    plt.ylim(-0.05, 1.05)
    for ap, name, color, style in zip(aps, names, colors, styles):
        plt.plot(recs, ap, label=name, c=color, linestyle=style)
    plt.legend()
    plt.legend(prop={'size': 12})
    plt.savefig(os.path.join(out_dir, '{}_full.pdf'.format(file_name)), 
        format='pdf', dpi=100)

def nms_wrapper(js_data, iou_thresh, diff_thresh):
    if len(js_data) == 0:
        return []
    image_id, boxes = js_data[0]['image_id'], []
    res = []
    img_ids = []
    for js in js_data:
        img_id = js['image_id']
        if img_id == image_id:
            boxes.append(js)
        else:
            nms_boxes = nms(boxes, iou_thresh, diff_thresh)
            boxes = [js]
            image_id = img_id
            res += nms_boxes
    return res

def nms(detections, thresh, diff_thresh):
    """Pure Python NMS baseline."""
    dets = np.array([ x['bbox'] for x in detections])
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]+x1
    y2 = dets[:, 3]+y1
    scores = np.array([x['score'] for x in detections])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        current_score = scores[i]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        over_inds = np.where(ovr > thresh)[0]
        if len(over_inds) == 0:
            keep.append(detections[i])
        else:
            next_highest_score = scores[order[over_inds[0]+1]]
            if current_score - next_highest_score > diff_thresh:
                keep.append(detections[i])
        order = order[inds + 1]            

    return keep

    