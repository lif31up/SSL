import torch
from torch.utils.data import DataLoader
from Embedder import load_MNIST
from Masker import Masker
from config import BaseConfig, BASE_SAVE_TO
from model.ViTBase import ViTBase
from train import train
from utils import get_transform_MNIST


if __name__ == "__main__":
  config = BaseConfig()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  mnist_transform = get_transform_MNIST(input_size=90)
  traindata, _ = load_MNIST(path='./data', transform=mnist_transform, len=(10000, 1000))
  trainset = Masker(dataset=traindata, config=config).consolidate()
  config.dummy = trainset.__getitem__(0)[0]
  trainloader = DataLoader(dataset=trainset, batch_size=config.batch_size)
  model = ViTBase(config=config)
  train(model=model, path=BASE_SAVE_TO, config=config, trainset=trainloader, device=device)
# if __name__ == "__main__":