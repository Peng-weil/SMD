import os
import io
import re
import sys


import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch import optim

from .utils import to_cuda
from .dataset import LLIMDataset

logger = getLogger()


def create_train_iterator(params, env, data_path):
    """
    Create a dataset for this environment.
    """
    logger.info(f"Creating train dataset ...")

    dataset = LLIMDataset(env=env, train=True, rng=None, params=params, path=data_path[0])

    return DataLoader(dataset,
                      timeout=(0 if params.num_workers == 0 else 1800),
                      batch_size=params.batch_size,
                      num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
                      shuffle=False,
                      collate_fn=dataset.collate_fn)


class Trainer(object):

    def __init__(self, modules, env, params):
        """
        Initialize trainer.
        """
        # modules / params
        self.modules = modules
        self.params = params
        self.env = env

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # experiment result
        self.result_log = []

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(self.modules[k], device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True)

        # set optimizers
        self.set_optimizer()

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 1
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict([('cross_entropy', [])])
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # reload exported data
        assert params.reload_data is not None
        self.data_path = params.reload_data.split(',')

        # create data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = iter(create_train_iterator(params, env, self.data_path))

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters['model'] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}
        s = params.optimizer

        if "," in s:
            method = s[:s.find(',')]
            optim_params = {}
            for x in s[s.find(',') + 1:].split(','):
                split = x.split('=')
                assert len(split) == 2
                assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
                optim_params[split[0]] = float(split[1])
        else:
            method = s
            optim_params = {}

        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)

        self.optimizers['model'] = optim.Adam(self.parameters['model'], **optim_params)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        print(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        for k, v in self.modules.items():
            v.load_state_dict(data[k])

        # reload optimizers
        for name in self.optimizers.keys():
            logger.warning(f"Reloading checkpoint optimizer {name} ...")
            self.optimizers[name].load_state_dict(data[f'{name}_optimizer'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def enc_dec_step(self):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.train()
        decoder.train()

        # batch
        (x1, len1), (x2, len2) = self.get_batch()

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

        # forward / loss
        encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
        decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
        _, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats['cross_entropy'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm > 0:
            for name in names:
                clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
        for optimizer in optimizers:
            optimizer.step()

        self.n_equations += params.batch_size

    def get_batch(self):
        try:
            batch = next(self.dataloader)
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} when fetching batch. "
                         "Arguments:{2!r}. Restarting ...".format(type(e).__name__,
                                                                  sys.exc_info()[-1].tb_lineno, e.args))

        return batch

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 20 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter

        for k, v in self.stats.items():
            loss_item = k
            loss_value = np.mean(v)
        s_stat = "{}: {:.2f} - ".format(loss_item.upper().replace('_', '-'), loss_value)

        for k, v in self.optimizers.items():
            for group in v.param_groups:
                s_lr = "LR: {:.4e}".format(group['lr'])

        new_time = time.time()
        diff = new_time - self.last_time
        self.last_time = new_time

        logger.info(f"{s_iter} {s_stat} {s_lr}")

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
            'params': {k: v
                       for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.optimizers[name].state_dict()

        torch.save(data, path)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        self.result_log.append(scores)

        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and self.params.is_master:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)." % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                logger.info(f'acc:{self.result_log}')
                exit()
        self.save_checkpoint('checkpoint')
        self.epoch += 1
