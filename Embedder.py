import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import torchvision as tv
from utils import get_transform_MNIST_10


class Embedder(Dataset):
  def __init__(self, dataset, config):
    super(Embedder, self).__init__()
    self.dataset, self.config = dataset, config
    self.is_consolidated = False
  # __init__

  def __len__(self): return len(self.dataset)

  def __getitem__(self, item):
    if self.is_consolidated: return self.dataset[item][0], self.dataset[item][1]
    feature, label = self.dataset[item]
    patches = feature.unfold(1, 30, 30).unfold(2, 30, 30).permute(1, 2, 0, 3, 4)
    flatten_patches = torch.reshape(input=patches, shape=(9, -1))
    label = F.one_hot(torch.tensor(label), num_classes=10).float()
    return flatten_patches, label
  # __getitem__

  def consolidate(self):
    buffer = list()
    progression = tqdm(self)
    for feature, label in progression: buffer.append((feature, label))
    self.dataset, self.is_consolidated = buffer, True
    return self
  # consolidate
# Embedder

def load_MNIST_10(transform, path='./data', trainset_len=1000, testset_len=500):
  # trainset, testset are provided as torch.nn.utils.dataset
  trainset = tv.datasets.MNIST(root=path, train=True, download=True, transform=transform)
  trainset_indices = torch.randperm(trainset.__len__()).tolist()[:trainset_len]
  trainset = Subset(dataset=trainset, indices=trainset_indices)
  testset = tv.datasets.MNIST(root=path, train=False, download=True, transform=transform)
  testset_indices = torch.randperm(testset.__len__()).tolist()[:testset_len]
  testset = Subset(dataset=testset, indices=testset_indices)
  return trainset, testset
# load_MNIST_10


if __name__ == "__main__":
  from config import Config
  config = Config()

  # load dataset, transform from folder
  cifar_10_transform = get_transform_MNIST_10(input_size=90)
  trainset, testset = load_MNIST_10(path='./data', transform=cifar_10_transform)

  # embed dataset (3 times 3 patches)
  trainset = Masker(dataset=trainset, config=config).consolidate()
  trainset = DataLoader(dataset=trainset, batch_size=config.batch_size)
  for feature, label in trainset:
    print(f'batch_size: {config.batch_size}\nfeature: {feature.shape}\nlabel:{label.shape}')
    break
  # for
# if __name__ == "__main__"