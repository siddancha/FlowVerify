"""
Example command:

python eval.py --config=../configs/tdid_avd2_easy_15vp.yaml\
               --run-dir=/scratch/sancha/osid/eval/01-19-19/15vp\
               --gt-json=../gen/gt/AVD_split2_easy_test.json\
               --dt-json=../gen/dt/TDID_AVD2_easy_OUT.json
"""

import argparse
from contextlib import redirect_stdout
import copy
import json
import numpy as np
import pickle
import os

from pipeline.eval_detector import MapEvaluator
import matplotlib.pyplot as plt


def get_json_datas_and_names_from_dir(dir):
    list_score_json_data, list_score_name = [], []
    for json_file in os.listdir(dir):
        if '.json' not in json_file: continue
        with open(os.path.join(dir, json_file), 'r') as f:
            score_json_data = json.load(f)
        list_score_json_data.append(score_json_data)

        score_name = os.path.basename(json_file)[:-5]
        list_score_name.append(score_name)

    return list_score_json_data, list_score_name


def merge_scores_into_json(dt_json_data, score_json_data, use_det_scores=False):
    # Dictionarize score_json
    score_json_dict = {entry['id']: entry for entry in score_json_data}

    merged_json_data = []
    for dt_json_entry in dt_json_data:
        dt_id, dt_score = dt_json_entry['id'], dt_json_entry['score']
        score_json_score = max(score_json_dict[dt_id]['scores']) # max scores across viewpoints.

        merged_json_entry = dt_json_entry.copy()
        if use_det_scores:
            if score_json_score > 0.5:
                merged_json_entry['score'] = dt_score
            else:
                merged_json_entry['score'] = 0.0
        else:
            merged_json_entry['score'] = score_json_score

        merged_json_data.append(merged_json_entry)

    return merged_json_data


# def _do_score_eval(run_dir, score_name, eval, dt_json_data, results_dir, min_ap_0=-1):
#     pr_data_dir = os.path.join(results_dir, 'pr_data')
#     if not os.path.isdir(pr_data_dir): os.makedirs(pr_data_dir)

#     print('Evaluating results for {} score ...'.format(score_name))

#     # Printing scores.
#     line = '-' * 80
#     print('\n' * 3)
#     print(line)
#     print('Evaluation for {} score.'.format(score_name))
#     print(line)
#     eval.do_eval_and_print(dt_json_data)
#     print(line)

#     # Plot PR curve
#     ap, ar = eval.get_avg_precision_recall()
#     params = eval.coco_eval.params

#     if ap[0] >= min_ap_0:
#         plt.plot(params.recThrs, ap, label=score_name)

#     pr_data_file = os.path.join(pr_data_dir, '{}.pkl'.format(score_name))
#     with open(pr_data_file, 'wb') as f:
#         pickle.dump({'ap': ap, 'ar': params.recThrs}, f)

#     iouThr, maxDet, nCats = params.iouThrs[0], params.maxDets[-1], len(params.catIds)
#     return ap, ar, iouThr, maxDet, nCats


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Evaluation script')
#     parser.add_argument('--config', default='config', help='config file path')
#     parser.add_argument('--run-dir', type=str, help='output directory of run.py')
#     parser.add_argument('--gt-json', type=str, default=None, help='json file containing GT results')
#     parser.add_argument('--dt-json', type=str, default=None, help='json file containing results of first stage')
#     parser.add_argument('--plot-better-than-detector', dest='plot_better_than_detector', action='store_true',
#                         help='plot PR curves for only those scores that have better precision at 0recall than detector')
#     parser.set_defaults(pbplot_better_than_detector=False)
#     args = parser.parse_args()

#     results_dir = os.path.join(args.run_dir, 'results')
#     if not os.path.isdir(results_dir): os.makedirs(results_dir)

#     with open(args.dt_json, 'r') as f:
#         dt_json_data = json.load(f)

#     list_score_json_data, list_score_name = get_json_datas_and_names_from_dir(args.run_dir)
#     combined_dir = os.path.join(args.run_dir, 'combined')
#     if os.path.isdir(combined_dir):
#         combined_list_score_json_data, combined_list_score_name = get_json_datas_and_names_from_dir(combined_dir)
#         list_score_json_data.extend(combined_list_score_json_data)
#         list_score_name.extend(combined_list_score_name)

#     results_file = os.path.join(results_dir, 'results.txt')
#     eval = MapEvaluator(args.gt_json)

#     with open(results_file, 'w') as f:
#         with redirect_stdout(f):
#             # Results for original detector.
#             ap, ar, iouThr, maxDet, nCats = _do_score_eval(
#                 args.run_dir, 'detector', eval, copy.deepcopy(dt_json_data), results_dir)
#             detector_ap_0 = ap[0]

#             # Results for each existing score json files.
#             for score_name, score_json_data in zip(list_score_name, list_score_json_data):
#                 use_det_scores = (len(score_name) > 4 and score_name[-4:] == '.UDS')
#                 merged_json_data = merge_scores_into_json(dt_json_data, score_json_data, use_det_scores=use_det_scores)
#                 min_ap_0 = detector_ap_0 if args.plot_better_than_detector else -1
#                 _do_score_eval(args.run_dir, score_name, eval, merged_json_data, results_dir, min_ap_0=min_ap_0)

#             # Results for random scores.
#             random_json_data = []
#             for data_entry in dt_json_data:
#                 random_entry = data_entry.copy()
#                 random_entry['score'] = np.random.uniform()
#                 random_json_data.append(random_entry)
#             _do_score_eval(args.run_dir, 'random', eval, random_json_data, results_dir)

#             plot_file_name = os.path.join(results_dir, 'pr_curves.pdf')
#             plt.ylim(-0.05, 1.05)
#             title = 'AP over {} classes for iouThr={:.2f}, maxDet={}. Mean max recall is {:.3f}.'.format(nCats, iouThr, maxDet, ar)
#             plt.title(title, fontsize=10)
#             plt.legend()
#             plt.legend(prop={'size': 3})
#             plt.savefig(plot_file_name, format='pdf', dpi=1000);
