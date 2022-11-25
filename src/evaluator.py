from logging import getLogger
from collections import OrderedDict
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import to_cuda
from .dataset import LLIMDataset

logger = getLogger()


def create_test_iterator(params, env, data_path):

    dataset = LLIMDataset(env=env, train=False, rng=np.random.RandomState(0), params=params, path=data_path[1])

    return DataLoader(dataset, timeout=0, batch_size=params.batch_size, num_workers=4, shuffle=False, collate_fn=dataset.collate_fn)


class Evaluator(object):
    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': self.trainer.epoch})

        with torch.no_grad():
            if self.params.eval_only:
                self.enc_dec_step(scores)
            elif self.params.gen_only:
                self.enc_dec_step_gen()
            else:
                self.enc_dec_step(scores)
        return scores

    def enc_dec_step(self, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.eval()
        decoder.eval()

        # stats
        xe_loss = 0
        correct_num = 0

        # iterator
        iterator = create_test_iterator(params=params, env=self.env, data_path=self.trainer.data_path)

        eval_size = len(iterator.dataset)

        i = 1
        for (x1, len1), (x2, len2) in iterator:
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
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)
            
            t = torch.zeros_like(pred_mask, device=y.device)
            pred_res = word_scores.max(1)[1]
            # temp1 = word_scores.max(1)[1]
            # temp2 = word_scores.max(1) 
            t[pred_mask] += pred_res == y
            # temp3 = t.T
            res = (t.T)[:, 0:-2]

            valid = (res.sum(1) == len2 - 2).cpu().long()
            # stats
            correct_num += valid.sum().item()
            xe_loss += loss.item() * len(y)

        # log
        logger.info(
            f"{correct_num}/{eval_size} ({100. * correct_num / eval_size}%) matrix were evaluated correctly.")

        scores[f'test_xe_loss'] = xe_loss / eval_size
        scores[f'test_acc'] = 100. * correct_num / eval_size
    
    def enc_dec_step_gen(self,data_type="test"):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.eval()
        decoder.eval()

        iterator = create_test_iterator(data_type=data_type, params=params, env=self.env,
                                        data_path=self.trainer.data_path)

        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2) in iterator:
            # target words to predict
            
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            # forward / loss
            encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
            str1, str2 = decoder('generate', src_enc=encoded, src_len=x1)
            
            logger.info(str1)
            logger.info(str2)

def convert_to_text(batch, lengths, id2word, params):
    """
    Convert a batch of sequences to a list of text sequences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sequences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(id2word[batch[k, j]])
        sequences.append(" ".join(words))
    return sequences
