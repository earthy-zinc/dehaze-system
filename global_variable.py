import os

import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

DEVICE_ID = [0]

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_PATH, "data")

CACHE_PATH = os.path.join(PROJECT_PATH, "cache")

MODEL_PATH = os.path.join(PROJECT_PATH, "trained_model")
