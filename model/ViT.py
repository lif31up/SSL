import torch
from torch import nn
from torch.utils.data import DataLoader
from Embedder import load_MNIST_10, Embedder
from config import Config
from utils import get_transform_MNIST_10


class ViT(nn.Module):
  def __init__(self, config):
    super(ViT, self).__init__()
    self.config = config
    self.stacks = nn.ModuleList([EncoderStack(self.config) for _ in range(config.n_stacks)])
    self.flatten = nn.Flatten(start_dim=1)
    self.cls = nn.Parameter(torch.zeros(config.dim))
    self.fc = self._get_fc(self.config.dummy).apply(self.config.init_weights)
  # __init__

  def add_cls(self, x):
    cls = self.cls.expand(x.shape[0], 1, -1)
    x = torch.cat([x, cls], dim=1)
    return x
  # add_cls

  def forward(self, x):
    x = self.add_cls(x)
    for stack in self.stacks: x = stack(x)
    return self.fc(self.flatten(x))
  # forward

  def _get_fc(self, dummy):
    with torch.no_grad():
      cls = self.cls.expand(1, -1)
      dummy = torch.cat([dummy, cls], dim=0)
      for stack in self.stacks: dummy = stack(dummy)
    dummy = dummy.flatten(start_dim=0)
    return nn.Linear(dummy.shape[0], self.config.output_dim, bias=self.config.bias)
  # _get_fc
# Transformer

class EncoderStack(nn.Module):
  def __init__(self, config):
    super(EncoderStack, self).__init__()
    self.config = config
    self.mt_attn = MultiHeadAttention(config, mode="scaled")
    self.ffn = nn.ModuleList()
    for _ in range(config.n_hidden):
      self.ffn.append(nn.Linear(config.dim, config.dim, bias=config.bias))
    self.activation, self.ln = nn.GELU(), nn.LayerNorm(config.dim)
    self.dropout = nn.Dropout(config.dropout)

    self.apply(self.config.init_weights)
  # __init__

  def forward(self, x):
    res = x
    x = self.ln(self.mt_attn(x) + res)
    res = x
    for i, layer in enumerate(self.ffn):
      if i != len(self.ffn): x = self.dropout(self.activation(layer(x)))
      else: x = self.dropout(layer(x))
    return self.ln(x + res)
  # forward
# EncoderStack

class MultiHeadAttention(nn.Module):
  def __init__(self, config, mode="scaled"):
    super(MultiHeadAttention, self).__init__()
    assert config.dim % config.n_heads == 0, "Dimension must be divisible by number of heads"
    self.config = config
    self.sqrt_d_k, self.mode = (config.dim // config.n_heads) ** 0.5, mode
    self.w_q, self.w_k = nn.Linear(config.dim, config.dim, bias=config.bias), nn.Linear(config.dim, config.dim, bias=config.bias)
    self.w_v, self.w_o = nn.Linear(config.dim, config.dim, bias=config.bias), nn.Linear(config.dim, config.dim, bias=config.bias)
    self.ln, self.dropout, self.softmax = nn.LayerNorm(config.dim), nn.Dropout(config.attention_dropout), nn.Softmax(dim=1)

    self.apply(self.config.init_weights)
  # __init__

  def forward(self, x, y=None):
    Q = self.w_q(x)
    (K, V) = (self.w_k(x), self.w_v(x)) if self.mode != "cross" else (self.w_k(y), self.w_v(y))
    raw_attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    down_scaled_raw_attn_scores = raw_attn_scores / self.sqrt_d_k
    if self.mode == "masked":
      masked_indices = torch.rand(*down_scaled_raw_attn_scores.shape[:-1], 1) < self.config.mask_prob
      down_scaled_raw_attn_scores[masked_indices] = float("-inf")
    attn_scores = self.softmax(down_scaled_raw_attn_scores)
    attn_scores = self.dropout(attn_scores)
    return self.ln(torch.matmul(attn_scores, V) + x)
  # attn_score
# MultiHeadAttention

if __name__ == "__main__":
  config = Config()
  # load dataset, transform from folder
  mnist_10_transform = get_transform_MNIST_10(input_size=225)
  trainset, testset = load_MNIST_10(path='./data', transform=mnist_10_transform)

  # embed dataset (3 times 3 patches)
  trainset = Embedder(dataset=trainset, config=config).consolidate()
  config.dummy = trainset.__getitem__(0)
  trainset = DataLoader(dataset=trainset, batch_size=config.batch_size)

  # init model
  model = ViT(config=config)
  model.get_fc(dummy=config.dummy)

  print(f'model: {model}')
# if __name__ == "__main__":