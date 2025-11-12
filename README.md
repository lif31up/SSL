# Self-Supervised Learning for Baseline ViT
This implementation is inspired by:
[Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531) by Geoffrey Hinton, Oriol Vinyals, Jeff Dean.
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.



The Vision Transformer (ViT) attains excellent results when pretrained at sufficient scale and transferred to tasks with fewer datapoints. When pretrained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state-of-the-art image recognition benchmarks.

- **Task:** Image Recognition
- **Dataset:** MNIST

### Experiment on CoLab
<a href="https://colab.research.google.com/drive/19tnhhYqi4iBZ7nbZ9NJF6YQA3LlGW8JF?usp=sharing">
  <img alt="colab" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/2560px-Google_Colaboratory_SVG_Logo.svg.png" width="160"></img>
</a>

|               | **Pretrained ViT with MLPs** |
|---------------|------------------------------|
| **ACC (1000)** | `90.9%`                      |
### Requirements
To run the code on your own machine, `run pip install -r requirements.txt`.
```text
tqdm>=4.67.1
```

### Configuration
`confing.py` contains the configurations settings for the two model (adapter and ViT), including the number of heads, dimensions, learning rate, and other hyperparameters.

```python
BASE_SAVE_TO = "bins/baseViT.bin"
BASE_LOAD_FROM = "bins/baseViT.bin"
PRETRAINED_SAVE_TO = "bins/pretrainedViT.bin"
PRETRAINED_FROM = "bins/pretrainedViT.bin"

class BaseConfig:
  def __init__(self):
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

    self.mask_val = -1e-9
    self.mask_ratio = 768

class AdapterConfig(BaseConfig):
  def __init__(self):
    super().__init__()
    self.output_dim = 10
```

### Pretraining
`pretrain.py` is to pretrain the model on the MNIST dataset with SSL.

```python
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
```

### Training
`train.py` is to pretrain the model on the MNIST dataset with Transfer Learning.

```python
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
```

### Evaluation
`evaluate.py` is used to evaluate the trained model on the MNIST-10 dataset. It loads the model and embedder, processes the dataset, and computes the accuracy of the model.

```python
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
```