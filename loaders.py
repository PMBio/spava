import itertools
import pickle
import random

import torch
import numpy as np
from torch.utils.data import SequentialSampler

import ignite.distributed as idist

NUM_WORKERS = 0
BATCH_SIZE = 1


def get_sampler(dataset):
    sampler = SequentialSampler(dataset)
    return sampler


def get_data_loader(dataset):
    # sampler = get_sampler(dataset)
    # we need num_workers = 0 in order to be able to debug with PyCharm multithreaded torch code via SSH, this is a known issue
    use_cuda = torch.cuda.is_available()
    if NUM_WORKERS == 0:
        print(
            f'warning: num_workers = {NUM_WORKERS}, if you are not debugging with PyCharm consider increasing this number')
    kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if use_cuda else {}
    f = idist.auto_dataloader
    loader = f(dataset, batch_size=BATCH_SIZE, **kwargs) # , sampler=sampler,
    return loader


if __name__ == '__main__':
    from ds import RawMeanDataset, RawMean12, NatureBImproved, NatureBOriginal, TransformedMeanDataset

    splits = ['train', 'validation', 'test']
    # splits = ['validation', 'test']
    for split in splits:
        a, b, c, d, e = get_data_loader(RawMeanDataset(split)), \
                        get_data_loader(RawMean12(split)), \
                        get_data_loader(NatureBImproved(split)), \
                        get_data_loader(NatureBOriginal(split)), \
                        get_data_loader(TransformedMeanDataset(split))
        for f in [a, b, c, d, e]:
            print(f.__iter__().__next__().shape)
