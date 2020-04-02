import functools

import dynet as dy
import numpy as np

import trees

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
