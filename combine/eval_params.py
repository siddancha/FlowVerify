import os
import json
import argparse
import numpy as np
from time import time

from flowmatch.utils import load_config
from combine.utils import evaluate, nms_wrapper, plot_pr_curves
from combine.scores import HardAND
from combine.evaluator import Evaluator

COLORS = ['tab:blue', 'tab:red', 'tab:green']

LINESTYLE = ['-', '-', '--']

OUT_DIR = 'plots'

def make_plot(aps, names, recs, title):
    plot_pr_curves( aps, 
                    names, 
                    recs, 
                    'plots/', 
                    '{}'.format(title), 
                    COLORS[:len(aps)], 
                    LINESTYLE[:len(aps)])

def evaluate_detector(cfg, cat_ids, dt_json_pth):
    with open(dt_json_pth, 'r') as f:
        dt_json_data = json.load(f)
    det_ap, det_ar, recs = evaluate(os.path.join(cfg.root, cfg.gt_json), 
                    dt_json_data, 
                    False,
                    cat_ids=cat_ids)
    return det_ap, det_ar, recs

def generate_nms_score_files(dt_json_pth, iou_thresh, ratio_thresh, run_dir, 
                             nvp=15):
    with open(dt_json_pth, 'r') as f:
        dt_json_data = json.load(f)
    out_pth = os.path.join(run_dir, 
            'nms_{:.1f}_{:.1f}.json'.format(iou_thresh, ratio_thresh))
    
    # if not os.path.exists(out_pth):
    nms_keep = nms_wrapper(dt_json_data, iou_thresh, ratio_thresh)
    keep_ids = set([x['id'] for x in nms_keep])
    nms_scores = []
    for dt in dt_json_data:
        nms_scores.append({'id': dt['id'], 
            'scores': [1 if dt['id'] in keep_ids else 0] * nvp})
    
    with open(out_pth, 'w') as f:
        json.dump(nms_scores, f)

def evaluate_scorers(cfg, cat_ids, dt_pth, scorers, params_list, 
                     names, plot_title):
    assert(len(scorers) == len(names))

    aps = []
    for scorer, params in zip(scorers, params_list):
        if params is None:
            ap, ar, recs = evaluate_detector(cfg, cat_ids, dt_pth)
        else:
            evaluator = Evaluator(scorer=scorer, 
                              run_dir=os.path.join(cfg.root,cfg.run_dir), 
                              gt_json_file=os.path.join(cfg.root, cfg.gt_json), 
                              dt_json_file=os.path.join(cfg.root, cfg.dt_json),
                              use_det_score=True,
                              cat_ids=cat_ids)

            ap, ar, recs = evaluator.get_pr_curve(params)
        aps.append(ap)

    if len(aps) <= 3:
        make_plot(aps, names, recs, plot_title)

    print(['{}:({:.3f},{:.3f})'.format(name, np.mean(x), np.max(x)) for x,name in zip(aps, names)])

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='config file path')
    parser.add_argument('--eval_ds', default='gmu_test', help='dataset to evaluate')
    parser.set_defaults()

    args = parser.parse_args()

    if not os.path.exists(OUT_DIR):
      os.makedirs(OUT_DIR)

    eval_ds = args.eval_ds
    plot_title = '{}'.format(eval_ds)
    cfg = load_config(args.config).combine
    cfg[eval_ds].root = cfg.root

    dt_pth = os.path.join(cfg[eval_ds].root, cfg[eval_ds].dt_json)

    scorers, names, params_list = [], [], []
    for name in cfg.scorers.keys():
        if cfg.scorers[name].score_names == 'None':
            scorer = None
            params = None
        else:
            scorer = HardAND(cfg.scorers[name].score_names)
            params = cfg.scorers[name].params
            if 'nvps' in cfg.scorers[name]:
                scorer.nvps = cfg.scorers[name].nvps

        names.append(name)
        scorers.append(scorer)
        params_list.append(params)

    cfg = cfg[eval_ds]
    cat_ids = cfg.catIds

    evaluate_scorers(cfg, cat_ids, dt_pth, scorers, params_list, 
                     names, plot_title)

