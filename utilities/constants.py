import torch
import pickle

SEPERATOR = "=" * 25

# Taken from the paper
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.98
ADAM_EPSILON = 10e-9

LR_DEFAULT_START = 1.0
SCHEDULER_WARMUP_STEPS = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_END = 3
TOKEN_PAD = 1

with open('./music_vocab.pkl') as f:
    vcb = pickle.load(f)
    VOCAB_SIZE = len(vcb)

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int32

TORCH_LABEL_TYPE = torch.long

PREPEND_ZEROS_WIDTH = 4
