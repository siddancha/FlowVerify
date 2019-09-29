import os
import time
import numpy as np
from functools import reduce
import math

import torch
import torch.optim as optim

from flowmatch.losses import epe_loss


class Trainer:
    def __init__(self, exp_dir, cfg, net, trainstream, train_writer, validator=None, resume=None):
        assert torch.cuda.is_available(), 'Error: CUDA not found!'
        self.start_iter = 0

        # Set seed
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)

        # Make exp folders
        self.ckpt_dir = os.path.join(exp_dir, 'ckpts')
        if not os.path.isdir(self.ckpt_dir): os.makedirs(self.ckpt_dir, exist_ok=True)

        self.cfg = cfg
        self.net = net
        self.trainstream = trainstream

        # Optimizer.
        if cfg.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
        else:
            raise ValueError('cfg.optimizer should be one of [\'sgd\', \'adam\']')

        if resume is not None:
            ckpt_path = os.path.join(self.ckpt_dir, 'ckpt_{}.pth'.format(resume))
            print('Resuming from checkpoint {} ...'.format(resume))
            checkpoint = torch.load(ckpt_path)
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # iter is measured in number of examples
            self.start_iter = checkpoint['iter']
            # self.trainstream.load_state(checkpoint['ds_state'])

        # # Parallelize model across GPUs.
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self.train_writer = train_writer
        self.validator = validator

    def _train_on_next_batch(self, collect_summaries):
        self.net.train()

        cs_input, tg_input, tg_mask, flow_target, img_pths = self.trainstream.next(self.cfg.batch_size)
        cs_input, tg_input, tg_mask, flow_target = cs_input.cuda(), tg_input.cuda(), tg_mask.cuda(), flow_target.cuda()

        self.optimizer.zero_grad()
        pred_flows_and_strides, summaries = self.net(cs_input, tg_input, collect_summaries)

        upsampled_pred_flows = [self.net.upsample(flow, scale_factor=stride) for flow, stride in pred_flows_and_strides]

        losses = epe_loss(upsampled_pred_flows, flow_target, tg_mask)
        loss = reduce(torch.add, losses) / len(losses)
        if torch.isnan(loss):
            print('nan encountered at {}'.format(img_pths))
            return loss.item(), summaries

        loss.backward()
        self.optimizer.step()

        return loss.item(), summaries

    def _train_subepoch(self, num_batches):
        """Trains a subepoch which is defined by num_batches"""
        sum_loss = 0
        skipped = 0
        for batch_idx in range(num_batches):
            is_last_batch = (batch_idx == num_batches - 1)
            batch_loss, summaries = self._train_on_next_batch(collect_summaries=is_last_batch)
            if math.isnan(batch_loss):
                skipped += 1
            else:
                sum_loss += batch_loss
            print('train_loss: %.3f | avg_loss: %.3f' % (batch_loss, sum_loss / (batch_idx + 1 - skipped)))

        return sum_loss / num_batches, summaries

    def _save_checkpoint(self, ckpt_path, loss, iter):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss,
            'iter': iter,
            'ds_state': self.trainstream.state()
        }
        print('Saving checkpoint to {} ...'.format(ckpt_path))
        torch.save(state, ckpt_path)

    def train(self):
        """Main train function"""
        ckpt_every = self.cfg.ckpt_every // self.cfg.batch_size  # in terms of batches
        num_batches = (self.cfg.num_iter - self.start_iter) // self.cfg.batch_size
        for batch_idx in range(ckpt_every, num_batches + 1, ckpt_every):
            batch_end_iter = self.start_iter + batch_idx * self.cfg.batch_size
            batch_start_iter = batch_end_iter - ckpt_every * self.cfg.batch_size + 1

            for lr_change_iter in self.cfg.lr_changes:
                if batch_start_iter <= lr_change_iter + 1 <= batch_end_iter:
                    new_lr = self.cfg.lr_changes[lr_change_iter]
                    for param_group in self.optimizer.param_groups:
                        print("Switched learning rate from {} to {}.\n".format(param_group['lr'], new_lr))
                        param_group['lr'] = new_lr

            print('Training from iter {} to {} (batch size {}) ...'.format(batch_start_iter, batch_end_iter,
                                                                           self.cfg.batch_size))
            start = time.time()
            loss, act_summaries = self._train_subepoch(ckpt_every)
            subepoch_time = time.time() - start

            print('\n------- Sub-Epoch (from {} to {}) -------'.format(batch_start_iter, batch_end_iter))
            print('Total time: {:.1f}s, speed: {:.3f}s/iter'.format(subepoch_time,
                                                                    subepoch_time / (ckpt_every * self.cfg.batch_size)))
            ckpt_path = os.path.join(self.ckpt_dir, 'ckpt_{}.pth'.format(batch_end_iter))
            self._save_checkpoint(ckpt_path, loss, batch_end_iter)

            # Tensorboard logging: loss summary.
            self.train_writer.add_scalar('train_loss', loss, batch_end_iter)

            # Tensorboard logging: weight and gradient summaries.
            # Gradient summaries are computed for the last batch that was trained on.
            for tag, tensor in self.net.named_parameters():
                self.train_writer.add_histogram('WEIGHT/' + tag, tensor.detach().cpu().numpy(), batch_end_iter)
                self.train_writer.add_histogram('GRAD/' + tag, tensor.grad.cpu().numpy(), batch_end_iter)

            # Tensorboard logging: activation summaries.
            for tag, np_array in act_summaries:
                self.train_writer.add_histogram('ACT/' + tag, np_array, batch_end_iter)

            if self.validator is not None:
                self.validator.run_validation(self.net, batch_end_iter)

            retr_succ, retr_fail = self.trainstream.num_retr_success, self.trainstream.num_retr_failure
            print('Failed to retrieve training examples {}/{} ({:.2f}%) times.'.format(
                retr_fail, retr_succ + retr_fail, retr_fail / (retr_succ + retr_fail) * 100.))
            print('-----------------------------------------\n')

        self.train_writer.close()
        if self.validator is not None: self.validator.close()
