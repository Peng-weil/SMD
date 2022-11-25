import io
from logging import getLogger

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from src.utils import get_sorting_result

logger = getLogger()


class LLIMDataset(Dataset):

    def __init__(self, env, train, rng, params, path):
        self.env = env
        self.rng = rng
        self.train = train
        self.sorting_type = params.sorting_type
        self.task = params.task
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank

        assert (train is True) == (rng is None)

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        # generation, or reloading from file
        logger.info(f"Loading data from {path} ...")
        with io.open(path, mode='r', encoding='utf-8') as f:
            if not train:
                lines = [line.rstrip() for line in f]
            else:
                lines = []
                for i, line in enumerate(f):
                    if i == params.reload_size:
                        break
                    if i % params.n_gpu_per_node == params.local_rank:
                        lines.append(line.rstrip())

        self.data = [xy.split('\t') for xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        logger.info(f"Loaded {len(self.data)} data from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test (default of 5000 if no file provided)
        if self.train:
            params.data_size = len(self.data)
            self.size = 1 << 60
        else:
            self.size = 5000 if path is None else len(self.data)

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)

        x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]

        x, x_len = self.batch_sequences(x)
        y, y_len = self.batch_sequences(y)

        return (x, x_len), (y, y_len)

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.env.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.env.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.env.eos_index

        return sent, lengths

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.rng is None:
            assert self.train is True
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.rng = np.random.RandomState([worker_id, self.global_rank, self.env_base_seed])
            logger.info(f"Initialized random generator for worker {worker_id}, with seed {[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed}).")

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        return self.read_sample(index)
    
    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            index = self.rng.randint(len(self.data))
        x, y = self.data[index]

        x = x.split()
        y = y.split()

        assert self.sorting_type == "SMR" or self.sorting_type == "SMC" or self.sorting_type == "SMD" or self.sorting_type == "counter-SMD"
        SORTING_METHOD = self.sorting_type
        TASK = self.task
        x = get_sorting_result(x, SORTING_METHOD)
        if TASK == "transpose" or TASK == "permutation":
            y = get_sorting_result(y, SORTING_METHOD)
        elif TASK == "MIS":
            y = y

        assert len(x) >= 1 and len(y) >= 1
        return x, y
