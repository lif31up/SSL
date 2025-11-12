import numpy as np
import torch
from Embedder import Embedder


class Masker(Embedder):
  def __getitem__(self, item):
    if self.__is_consolidated: return self.dataset[item][0], self.dataset[item][1]
    feature, label = self.dataset[item]
    patches = feature.unfold(1, 30, 30).unfold(2, 30, 30).permute(1, 2, 0, 3, 4)
    flatten_patches = torch.reshape(input=patches, shape=(9, -1))
    masked_flatten_patches = self.mask(patches=flatten_patches)
    return masked_flatten_patches, patches.flatten()
  # __getitem__

  def mask(self, patches):
    for patch in patches[1:]:
      indices = np.random.randint(low=0, high=899, size=self.config.mask_ratio)
      patch[indices] = self.config.mask_val
    return patches
  # mask
# Masker