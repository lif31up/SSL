from torch import nn

# path vars
TEACH_SAVE_TO = "./vit.bin"
TEACH_LOAD_FROM = "./vit.bin"
STNDT_SAVE_TO = "./student.bin"
STNDT_LOAD_FROM = "./student.bin"

class Config:
  def __init__(self, is_teacher=False):
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

    self.masking_value = -1e-9
  # __init__
# Config

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
# init_weights