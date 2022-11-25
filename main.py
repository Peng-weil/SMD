import argparse
import json
import os
import random
from unittest.mock import patch

import numpy as np
import torch

import src.utils
from src.distribution import init_distributed_mode
from src.evaluator import Evaluator
from src.LLIM_env import LLIM_ENV
from src.model import build_modules
from src.trainer import Trainer
from src.utils import bool_flag, init_exp

np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Learn Laws in Matrix")

    # main parameters
    parser.add_argument('--dump_path', type=str, default='./dumped/')
    parser.add_argument('--exp_name', type=str, default='transpose')
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--sorting_type', type=str, default='counter-SMD', help="SMR,SMC,SMD,counter-SMD")
    parser.add_argument('--task', type=str, default="transpose")

    # model parameters
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--n_enc_layers', type=int, default=6)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--attention_dropout', type=float, default=0)
    parser.add_argument('--share_inout_emb', type=bool_flag, default=True)
    parser.add_argument('--sinusoidal_embeddings', type=bool_flag, default=False)

    # training parameters
    parser.add_argument("--env_base_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001")
    parser.add_argument("--clip_grad_norm", type=float, default=5)
    parser.add_argument("--epoch_size", type=int, default=20000)
    parser.add_argument("--max_epoch", type=int, default=200000)
    parser.add_argument("--stopping_criterion", type=str, default="test_acc,200")
    parser.add_argument("--validation_metrics", type=str, default="test_acc")
    parser.add_argument('--num_workers', type=int, default=0)

    # export data / reload it
    parser.add_argument('--reload_data', type=str, default=' ,dataset/eva/transpose/transpose_2_50000.test')
    parser.add_argument('--reload_size', type=int, default=-1)

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="")
    parser.add_argument("--reload_checkpoint", type=str, default="")

    # evaluation
    parser.add_argument('--eval_only', type=bool_flag, default=True)
    parser.add_argument('--gen_only', type=bool_flag, default=False)

    # CPU / multi-gpu / multi-node
    parser.add_argument('--cpu', type=bool_flag, default=False)
    parser.add_argument('--NGPU', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--master_port', type=int, default=-1)

    return parser


def main(params):
    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = init_exp(params)

    # CPU / CUDA
    if params.cpu:
        assert not params.multi_gpu
    else:
        assert torch.cuda.is_available()
    src.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    env = LLIM_ENV(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    # evaluation
    if params.eval_only:
        assert not params.gen_only
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    if params.gen_only:
        evaluator.run_all_evals()
        exit()

    # training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0

        while trainer.n_equations < trainer.epoch_size:
            trainer.enc_dec_step()
            trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals()

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.end_epoch(scores)

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    
    main(params)
