import random
import numpy as np

# torch
import torch

def set_seed(args) :
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)