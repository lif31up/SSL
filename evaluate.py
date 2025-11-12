import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Embedder import Embedder, load_MNIST
from config import BaseConfig, BASE_LOAD_FROM, PRETRAINED_FROM, AdapterConfig
from model.Adapter import Adapter
from model.ViTBase import ViTBase
from utils import get_transform_MNIST


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
  base_config = BaseConfig()
  adapter_config = AdapterConfig()
  mnist_10_transform = get_transform_MNIST(input_size=90)
  _, testdata = load_MNIST(path='./data', transform=mnist_10_transform, len=(1, 1000))
  testset = Embedder(dataset=testdata, config=base_config).consolidate()
  base_config.dummy = testset.__getitem__(0)[0]
  testloader = DataLoader(dataset=testset, batch_size=base_config.batch_size)
  base_data = torch.load(f=BASE_LOAD_FROM, map_location=torch.device('cpu'), weights_only=True)
  base = ViTBase(base_config)
  base.load_state_dict(base_data['sate'])
  adapter_data = torch.load(f=PRETRAINED_FROM, map_location=torch.device('cpu'), weights_only=True)
  adapter = Adapter(adapter_config, base=base)
  evaluate(model=adapter, dataset=testloader, device=device)
# if __name__ == "__main__":