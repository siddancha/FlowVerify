'''
script to generate SIFT scores
'''
import argparse
import json
import numpy as np
import os
import shutil
import time
import torch
from tqdm import tqdm

from flowmatch.eval.networks.vis import draw_images

from flowmatch import test as flow_test
from baselines.sift import test as sift_test
from flowmatch.datasets import load_dataset
from flowmatch.networks.flownet_cc import FlowNetC
from flowmatch.utils import load_config
from baselines.sift.siftscorer import SiftMatchScorer


def run_evaluation(dataset, net, args):
    start = time.time()

    # Setting up directories.
    if os.path.isdir(args.out_dir): shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(args.out_dir, 'vis')
        vis_pos_dir = os.path.join(vis_dir, 'pos')
        vis_neg_dir = os.path.join(vis_dir, 'neg')
        for dir in [vis_dir, vis_pos_dir, vis_neg_dir]: os.makedirs(dir)

    score_names = SiftMatchScorer.score_names

    def get_scores(example):
        sift_matches = sift_test.test(net, example)
        example['pred_sift'] = sift_matches

        # Reclassification.
        reclass_scorer = SiftMatchScorer(example)
        scores = reclass_scorer.get_scores()
        return scores

    list_score_json_data = [[] for _ in score_names]
    list_score_json_files = [os.path.join(args.out_dir, '{}.json'.format(score_name))
                             for score_name in score_names]

    num_eval = min(len(dataset), args.N)
    num_remaining = num_eval
    print('Evaluating {} examples ...'.format(num_eval))

    indices = np.random.permutation(len(dataset.dt_ids)) if args.randomize_order else range(len(dataset.dt_ids))
    for index_ind, index in enumerate(indices):
        if num_remaining == 0: break

        dt_id = dataset.dt_ids[index]
        scene_example = dataset.get_scene_example_from_dt_id(dt_id)
        if scene_example is None:
            raise Exception('scene_example was None')

        category_id = scene_example['category_id']
        list_target_example = dataset.tg_store[category_id]  # contains a target example per viewpoint

        list_example = []  # each element corresponds to a viewpoint
        list_list_score = [[] for _ in score_names]  # each element corresponds to a score type
        for vp_id, target_example in tqdm(enumerate(list_target_example)):
            example = {**scene_example, **target_example}
            scores = get_scores(example)  # this modifies the example
            for i, score in enumerate(scores):
                list_list_score[i].append(score)

            list_example.append(example)

        # Max score based viewpoints.
        list_max_score_vp_index = []  # length is number of scores
        for i, list_score in enumerate(list_list_score):
            max_score_ind = np.argmax(list_score)
            list_score_json_data[i].append({'id': dt_id, 'scores': list_score})
            list_max_score_vp_index.append(max_score_ind)

        if num_remaining == 1 or args.save_every is not None and (index_ind+1) % args.save_every == 0:
            for i, score_name in enumerate(score_names):
                score_json_data, score_json_file = list_score_json_data[i], list_score_json_files[i]
                print('Saving {} reclass outputs to {} ...'.format(score_name, score_json_file))
                with open(score_json_file, 'w') as f:
                    json.dump(score_json_data, f)

        num_remaining -= 1
        num_done = num_eval - num_remaining
        print('{}/{} ({:.1f}%) detections done ...'.format(num_done, num_eval, (num_done / num_eval) * 100.))

    evaluation_time = time.time() - start
    print('Total evaluation time: {:.1f}s'.format(evaluation_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combined evaluation for second and third stage')
    parser.add_argument('--config', default='config', help='config file path')
    parser.add_argument('--N', type=int, default=None, help='number of evaluation examples')
    parser.add_argument('--randomize-order', dest='randomize_order', action='store_true', help='evaluate detections in random order')
    parser.add_argument('--out-dir', type=str, help='directory where evaluation results will be stored')
    parser.add_argument('--save-every', type=int, default=None, help='save json output after these many reclassification')
    parser.add_argument('--save-vis', dest='save_vis', action='store_true', help='save visualization images')
    parser.add_argument('--save-vis-prob', type=float, default=1., help='visualizations will be saved with this prob')
    parser.set_defaults(randomize_order=False, save_vis=False)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print('Loading ObjectDetectionDataset ...')
    dataset = load_dataset('object_detection', cfg)

    if args.N is None:
        args.N = len(dataset.dt_ids)

    print('Building model...')
    net = FlowNetC(batchNorm=False, cfg=cfg.flownet).cuda()

    run_evaluation(dataset, net, args)

