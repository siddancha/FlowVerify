from contextlib import redirect_stdout
import json
import numpy as np
import os

from pipeline.eval_detector import MapEvaluator
from pipeline.eval.eval import merge_scores_into_json
from combine.utils import evaluate, nms_wrapper, rerank_dets

class Evaluator:
    def __init__(self, scorer, run_dir, gt_json_file, dt_json_file, 
                 use_det_score, cat_ids=None):
        self.scorer = scorer
        self.run_dir = run_dir
        self.use_det_score = use_det_score

        self.gt_json_file = gt_json_file
        self.dt_json_file = dt_json_file
        with open(self.dt_json_file, 'r') as f:
            self.dt_json_data = json.load(f)

        self.list_score_json_dict = []
        for score_name in self.scorer.list_score_names:
            score_file = os.path.join(self.run_dir, '{}.json'.format(score_name))
            with open(score_file, 'r') as f:
                score_json_data = json.load(f)
                score_json_dict = {entry['id']: entry['scores'] for entry in score_json_data}
            self.list_score_json_dict.append(score_json_dict)

        # Check if all score json files have the same set of ids.
        for score_json_dict_1, score_json_dict_2 in zip(self.list_score_json_dict[:-1], self.list_score_json_dict[1:]):
            assert(set(score_json_dict_1) == set(score_json_dict_2))

        self.cat_ids = cat_ids
        with redirect_stdout(None):
            self.eval = MapEvaluator(self.gt_json_file, self.cat_ids)

    def combine_scores(self, params):
        combined_score_json_data = []
        for id in sorted(self.list_score_json_dict[0].keys()):
            # Scores arr is a VxN array where V is number of viewpoints and N is number of scores.
            scores_arr = np.array([score_json_dict[id] for score_json_dict in self.list_score_json_dict]).T
            combined_scores = [self.scorer.score(scores, params) for scores in scores_arr]
            combined_score_json_data.append({'id': id, 'scores': combined_scores})
        return combined_score_json_data

    def _get_merged_json_data(self, params):
        combined_score_json_data = self.combine_scores(params)
        merged_json_data = rerank_dets(self.dt_json_data, combined_score_json_data)
        return merged_json_data

    def _do_eval(self, params):
        # Merge and temporarily save combined_score_json_data.
        if len(params) > 0:
            merged_json_data = self._get_merged_json_data(params)
        else:
            merged_json_data = self.dt_json_data

        with redirect_stdout(None):
            self.eval.do_eval(merged_json_data)

        ap, ar = self.eval.get_avg_precision_recall()
        recs = self.eval.coco_eval.params.recThrs
        return ap, ar, recs

    def get_accuracy(self, params):
        ap, ar, recs = self._do_eval(params)
        return np.around(np.mean(ap), decimals=3)

    def get_pr_curve(self, params):
        ap, ar, recs = self._do_eval(params)
        return ap, ar, recs

