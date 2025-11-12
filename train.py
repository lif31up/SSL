import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from Embedder import load_MNIST, Embedder
from config import BaseConfig, AdapterConfig, PRETRAINED_SAVE_TO, BASE_LOAD_FROM
from evaluate import evaluate
from model.Adapter import Adapter
from model.ViTBase import ViTBase
from utils import get_transform_MNIST


def train(model:nn.Module, path: str, config, trainset, device):
  model.to(device)
  model.train()

  # optim, criterion, scheduler
  optim = torch.optim.Adam(model.parameters(), lr=config.lr, eps=config.eps)
  criterion = nn.CrossEntropyLoss()
  scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

  progression = tqdm(range(config.iters))
  for _ in progression:
    for feature, label in trainset:
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      pred = model(feature)
      loss = criterion(pred, label)
      optim.zero_grad()
      loss.backward()
      optim.step()
    # for feature label
    scheduler.step()
    progression.set_postfix(loss=loss.item())
  # for in progression

  features = {
    "sate": model.state_dict(),
    "config": config
  } # feature
  torch.save(features, f"{path}")
# train


# TRANSFER LEARNING ON BASELINE MODEL
if __name__ == "__main__":
  base_config = BaseConfig()
  adapter_config = AdapterConfig()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  mnist_10_transform = get_transform_MNIST(input_size=90)
  traindata, testdata = load_MNIST(path='./data', transform=mnist_10_transform, len=(10000, 1000))
  trainset = Embedder(dataset=traindata, config=base_config).consolidate()
  base_config.dummy = trainset.__getitem__(0)[0]
  trainloader = DataLoader(dataset=trainset, batch_size=base_config.batch_size)
  testset = Embedder(dataset=testdata, config=base_config).consolidate()
  testloader = DataLoader(dataset=testset, batch_size=base_config.batch_size)
  data = torch.load(f=f"{BASE_LOAD_FROM}", weights_only=False, map_location=device)
  base = ViTBase(base_config).load_state_dict(data['state'])
  model = Adapter(config=adapter_config, base=base)
  train(model=model, path=PRETRAINED_SAVE_TO, config=adapter_config, trainset=trainloader, device=device)
  evaluate(model=model, dataset=testloader, device=device)
# if __name__ == "__main__":