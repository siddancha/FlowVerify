import os
import numpy as np
import time

import flowmatch.test as test
from flowmatch.losses import epe_loss

from tensorboardX import SummaryWriter


class Validator:
    def __init__(self, datasets, tb_tags, cfg, tb_dir):
        """
        Args:
            datasets: list(dataset) list of datasets of len N.
            tb_tags: list(str) list of tensorboard tags for each dataset, of len N.
            cfg: config.
            tb_dir: (str) directory path where tensorboard logs are dumped.
        """
        assert(len(datasets) == len(tb_tags))
        self.datasets = datasets
        self.tb_tags = tb_tags
        self.cfg = cfg
        self.writers = [SummaryWriter(os.path.join(tb_dir, tb_tag)) for tb_tag in tb_tags]

    def run_validation(self, net, iteration):
        start = time.time()

        for i in range(len(self.datasets)):
            dataset, writer, tb_tag = self.datasets[i], self.writers[i], self.tb_tags[i]

            indices = iter(np.random.permutation(len(dataset)))
            num_valid = min(len(dataset), self.cfg.num_valid)

            # Calculate epe loss.
            losses = []
            while len(losses) < num_valid:
                index = next(indices)
                example = dataset[index]
                if example is None: continue 
                flow_pred = test.raw_net_output(net, example)

                # "example" has been pre-processed and new tensors have been added. Now add GT flow.
                example = dataset.add_gt_flow(example)
                # if this example fails, abandon it and continue to next example
                if example is None: continue 
                
                flow_target = example['net_flow'].unsqueeze(0).cuda()
                tg_mask = example['net_tg_mask'].unsqueeze(0).cuda()
                loss = epe_loss([flow_pred], flow_target, tg_mask)[0]

                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)

            # Printing and logging.
            print('Avg EPE loss on {} examples for {} - | {:.3f}'.format(
                self.cfg.num_valid, tb_tag, avg_loss))
            writer.add_scalar('avg_epe_loss', avg_loss, iteration)

        validation_time = time.time() - start
        print('Total validation time: {:.1f}s'.format(validation_time))

    def close(self):
        for writer in self.writers: writer.close()