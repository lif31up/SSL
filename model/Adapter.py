import torch
from torch import nn


class Adapter(nn.Module):
  def __init__(self, config, base):
    super(Adapter, self).__init__()
    self.base = base
    self.config = config
    self.flatten = nn.Flatten(start_dim=1)
    self.cls = nn.Parameter(torch.zeros(config.dim))
    self.fc = self._get_fc(self.config.dummy.to(torch.device('cpu'))).apply(self.config.init_weights)

  # __init__

  def add_cls(self, x):
    cls = self.cls.expand(x.shape[0], 1, -1)
    x = torch.cat([x, cls], dim=1)
    return x

  # add_cls

  def _get_fc(self, dummy):
    with torch.no_grad():
      cls = self.cls.expand(1, -1)
      dummy = torch.cat([dummy, cls], dim=0)
    dummy = dummy.flatten(start_dim=0)
    return nn.Linear(dummy.shape[0], self.config.output_dim, bias=self.config.bias)

  # _get_fc

  def forward(self, x):
    x = self.add_cls(x)
    x = self.base(x)
    return self.fc(self.flatten(x))
  # forward
# LRViT