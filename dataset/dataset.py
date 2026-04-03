"""
Dataset class for loading data from the dataset
"""

import torch
import numpy
import random

# Init function for dataset
def seed_worker(worker_id):
    # Generate a unique seed for each worker based on worker_id and the global seed
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
