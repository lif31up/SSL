from torch import nn


BASE_SAVE_TO = "bins/baseViT.bin"
BASE_LOAD_FROM = "bins/baseViT.bin"
PRETRAINED_SAVE_TO = "bins/pretrainedViT.bin"
PRETRAINED_FROM = "bins/pretrainedViT.bin"

class BaseConfig:
  def __init__(self):
    self.iters = 50
    self.batch_size = 16
    self.dataset_len, self.testset_len = 1000, 500
    self.dummy = None

    self.n_heads = 3
    self.n_stacks = 6
    self.n_hidden = 3
    self.dim = 900
    self.output_dim = 10
    self.bias = True

    self.dropout = 0.1
    self.attention_dropout = 0.1
    self.eps = 1e-3
    self.betas = (0.9, 0.98)
    self.epochs = 5
    self.batch_size = 16
    self.lr = 1e-4
    self.clip_grad = False
    self.mask_prob = 0.3
    self.init_weights = init_weights

    self.mask_val = -1e-9
    self.mask_ratio = 768
  # __init__
# Config

class AdapterConfig(BaseConfig):
  def __init__(self):
    super().__init__()
    self.output_dim = 10
  # __init__()
# AdapterConfig

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
# init_weights