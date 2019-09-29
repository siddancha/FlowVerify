import argparse
import json
import numpy as np
from multiprocessing import Process, Pipe
import os

from combine.evaluator import Evaluator
from combine.scores import HardAND, SoftAND
from flowmatch.utils import load_config

def _get_param_combinations(lists):
    """Recursive function which generates a list of all possible parameter values"""
    if len(lists) == 1:
        list_p_1 = [[e] for e in lists[0]]
        return list_p_1

    list_p_n_minus_1 = _get_param_combinations(lists[1:])
    list_p_1 = [[e] for e in lists[0]]

    list_p_n = [p_1 + p_n_minus_1 for p_1 in list_p_1 for p_n_minus_1 in list_p_n_minus_1]
    return list_p_n

def parallel_grid_search(params, evaluator):
    list_params = _get_param_combinations(params)
    new_list_acc, new_list_params = [], []

    print('GRID_SEARCH: performing grid search over {} parameter combinations ...'.format(len(list_params)))

    def worker(job_index, params, send_end):
        acc = evaluator.get_accuracy(params)
        send_end.send((acc, params))
        send_end.send(job_index)

    # Create and launch jobs for grid search.

    max_parallel_jobs = 400
    for finer in range(1):
        for start in range(0, len(list_params), max_parallel_jobs):
            end = start + max_parallel_jobs
            inner_list_params = list_params[start:end]

            print('GRID_SEARCH: Performing grid search over a current batch of {}/{} parameters ...'.format(
                end,
                len(list_params)
            ))

            inner_list_proc, inner_list_pipe = [], []

            for job_index, params in enumerate(inner_list_params):
                recv_end, send_end = Pipe(False)
                proc = Process(target=worker, args=(job_index, params, send_end))
                inner_list_proc.append(proc); inner_list_pipe.append(recv_end)
                proc.start()

            for proc in inner_list_proc:
                proc.join()

            inner_list_result = [pipe.recv() for pipe in inner_list_pipe]
            inner_list_acc, inner_list_params = zip(*inner_list_result)
            
            new_list_acc.extend(inner_list_acc)
            new_list_params.extend(inner_list_params)   

    # print optimal parameters
    eval_list_acc = np.array(new_list_acc)

    best_acc_ind = eval_list_acc.argmax()
    best_acc, best_params = eval_list_acc[best_acc_ind], new_list_params[best_acc_ind]
    
    print('------------------------------------')
    print('Maximum mAP: {:.3f} -- {}'.format(best_acc, best_params))
    print('------------------------------------')

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../pipeline/configs/tdid_ycb.yaml', help='config file path')
    parser.add_argument('--run-dir', type=str, help='directory where evaluation results will be stored')
    parser.add_argument('--sequential', default=False, help='set to false to enable multiprocessing')
    parser.set_defaults()

    args = parser.parse_args()

    cfg = load_config(args.config).combine
    scorer = HardAND(cfg.score_names)
    catIds = cfg.catIds

    evaluator = Evaluator(scorer=scorer, 
                          run_dir=os.path.join(cfg.root,cfg.run_dir), 
                          gt_json_file=os.path.join(cfg.root, cfg.gt_json), 
                          dt_json_file=os.path.join(cfg.root, cfg.dt_json),
                          use_det_score=cfg.use_det_score,
                          cat_ids=catIds)

    params = [np.linspace(*x) for x in cfg.grid]

    parallel_grid_search(params, evaluator)
