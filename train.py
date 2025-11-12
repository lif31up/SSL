import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from Embedder import load_MNIST_10, Embedder
from config import Config, TEACH_SAVE_TO
from model.ViT import ViT
from utils import get_transform_MNIST_10


def train(model:nn.Module, path: str, config: Config, trainset, device):
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

if __name__ == "__main__":
  config = Config()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # load dataset, transform from folder
  mnist_10_transform = get_transform_MNIST_10(input_size=90)
  trainset, testset = load_MNIST_10(path='./data', transform=mnist_10_transform)
  # embed dataset (3 times 3 patches)
  trainset = Embedder(dataset=trainset, config=config).consolidate()
  config.dummy = trainset.__getitem__(0)[0]
  trainset = DataLoader(dataset=trainset, batch_size=config.batch_size)
  testset = Embedder(dataset=testset, config=config).consolidate()
  testset = DataLoader(dataset=testset, batch_size=config.batch_size)
  model = ViT(config=config)
  train(model=model, path=TEACH_SAVE_TO, config=config, trainset=trainset, device=device)
# if __name__ == "__main__":