import argparse
import json
import numpy as np
import os
import pickle
import shutil
import time
import torch

from flowmatch.losses import epe_loss
from flowmatch.eval.networks.vis import draw_images

from flowmatch import test
from flowmatch.datasets import load_dataset
from flowmatch.networks.flownet_simple import FlowNetS
from flowmatch.networks.flownet_cc import FlowNetC
from flowmatch.utils import load_config


def run_evaluation(dataset, net, args):
    start = time.time()

    # Setting up directories.
    if os.path.isdir(args.out_dir): shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_flows:
        flows_dir = os.path.join(args.out_dir, 'flows')
        os.makedirs(flows_dir)
    if args.save_vis:
        vis_dir = os.path.join(args.out_dir, 'vis')
        os.makedirs(vis_dir)

    num_eval = min(len(dataset), args.N)
    num_remaining = num_eval
    losses, json_data = [], []
    print('Evaluating {} examples ...'.format(num_eval))

    indices = np.random.permutation(len(dataset)) if args.randomize_order else range(len(dataset))
    for index in indices:
        if num_remaining == 0: break

        example = dataset[index]
        id_str = dataset.id_to_str(dataset.ids[index])

        if example is None: continue
        flow_pred = test.raw_net_output(net, example)

        # "example" has been pre-processed and new tensors have been added. Now add GT flow.
        example = dataset.add_gt_flow(example)
        # if this example fails, abandon it and continue to next example
        if example is None: continue
        
        if args.save_losses:
            flow_target = example['net_flow'].unsqueeze(0).cuda()
            tg_mask = example['net_tg_mask'].unsqueeze(0).cuda()
            loss = epe_loss([flow_pred], flow_target, tg_mask)[0]

            if np.isnan(loss.item()): continue  # TODO: confirm with Jenny if this is required

            losses.append(loss.item())
            json_data.append({'id': id_str, 'epe_loss': loss.item()})
            print('EPE loss on example {} -| {:.3f}, Avg EPE loss - | {:.3f}'.format(
                index, loss.item(), sum(losses) / len(losses)))

        example['pred_flow'] = test.postprocess_raw_net_output(flow_pred, example)

        if args.save_vis and np.random.uniform() < args.save_vis_prob:
            draw_images(example, out_file=os.path.join(vis_dir, '{}.pdf'.format(id_str)))

        if args.save_flows:
            with open(os.path.join(flows_dir, '{}.pkl'.format(id_str)), 'wb') as f:
                pickle.dump(example['pred_flow'], f)

        num_remaining -= 1
        num_done = num_eval - num_remaining
        print('{}/{} ({:.1f}%) examples done ...'.format(num_done, num_eval, (num_done / num_eval) * 100.))

    if args.save_losses:
        avg_loss = sum(losses) / len(losses)

        # Logging and printing.
        json_data = {'avg_epe_loss': avg_loss, 'num_examples': num_eval, 'all_losses': json_data}
        with open(os.path.join(args.out_dir, 'losses.json'), 'w') as f:
            json.dump(json_data, f)
        print('Avg EPE loss on {} examples - | {:.3f}'.format(num_eval, avg_loss))

    evaluation_time = time.time() - start
    print('Total evaluation time: {:.1f}s'.format(evaluation_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--config', default='config', help='config file path')
    parser.add_argument('--model-path', type=str, default=None, help='path to model for testing')
    parser.add_argument('--dataset', type=str, default=None, help='evaluation dataset')
    parser.add_argument('--N', type=int, default=None, help='number of evaluation examples')
    parser.add_argument('--out-dir', type=str, help='directory where evaluation results will be stored')
    parser.add_argument('--randomize-order', dest='randomize_order', action='store_true', help='evaluate detections in random order')
    parser.add_argument('--save-vis', dest='save_vis', action='store_true', help='save visualization images')
    parser.add_argument('--save-vis-prob', type=float, default=1., help='visualizations will be saved with this prob')
    parser.add_argument('--save-flows', dest='save_flows', action='store_true', help='save flow fields')
    parser.add_argument('--save-losses', dest='save_losses', action='store_true', help='compute and save losses')
    parser.set_defaults(randomize_order=False, save_vis=False, save_flows=False, save_losses=False)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print('Loading dataset ...')
    dataset = load_dataset(args.dataset, cfg)

    if args.N is None:
        args.N = len(dataset)

    print('Building model...')
    if cfg.flownet.arch == 'FlowNetS':
        net = FlowNetS(input_channels=6, batchNorm=False, cfg=cfg.flownet).cuda()
    elif cfg.flownet.arch == 'FlowNetC':
        net = FlowNetC(batchNorm=False, cfg=cfg.flownet).cuda()
    else:
        raise Exception("cfg.flownet.arch must be one of [FlowNetC, FlowNetS]")

    state_dict = torch.load(args.model_path)['net']
    net.load_state_dict(state_dict)

    run_evaluation(dataset, net, args)

