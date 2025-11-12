import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Embedder import Embedder, load_MNIST_10
from config import Config, TEACH_LOAD_FROM, STNDT_LOAD_FROM
from model.ViT import ViT
from utils import get_transform_MNIST_10


def evaluate(model, dataset, device):
  model.to(device)
  model.eval()
  correct, n_total = 0, 0
  for feature, label in tqdm(dataset):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    output = model.forward(feature)
    output = torch.softmax(output, dim=-1)
    pred = torch.argmax(input=output, dim=-1)
    label = torch.argmax(input=label, dim=-1)
    for p, l in zip(pred, label):
      if p == l: correct += 1
      n_total += 1
  # for
  print(f"Accuracy: {correct / n_total:.4f}")
# eval

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  config = Config()
  # load dataset, transform from folder
  mnist_10_transform = get_transform_MNIST_10(input_size=90)
  trainset, testset = load_MNIST_10(path='./data', transform=mnist_10_transform)
  # embed dataset (3 times 3 patches)
  trainset = Embedder(dataset=trainset, config=config).consolidate()
  config.dummy = trainset.__getitem__(0)[0]
  trainset = DataLoader(dataset=trainset, batch_size=config.batch_size)
  testset = Embedder(dataset=testset, config=config).consolidate()
  testset = DataLoader(dataset=testset, batch_size=config.batch_size)
  model_data = torch.load(f=TEACH_LOAD_FROM, map_location=torch.device('cpu'), weights_only=False)
  model = ViT(config)
  model.load_state_dict(model_data['sate'])
  evaluate(model=model, dataset=testset, device=device)
# if __name__ == "__main__":