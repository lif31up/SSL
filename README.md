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
`confing.py` contains the configuration settings for the model, including the number of heads, dimensions, learning rate, and other hyperparameters.

```python
TEACH_SAVE_TO = "./vit.bin"
TEACH_LOAD_FROM = "./vit.bin"
STNDT_SAVE_TO = "./student.bin"
STNDT_LOAD_FROM = "./student.bin"

class Config:
  def __init__(self, is_teacher=False):
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
```

### Training and Pretraining
By editing `PRETRAINING` in `train.py`, you can select how to train the model on the MNIST dataset. It includes the training loop, evaluation, and saving the model checkpoints.

```python
PRETRAINING = True

# TRANSFER LEARNING ON BASELINE MODEL
if __name__ == "__main__" and not PRETRAINING:
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
  evaluate(model=model, dataset=testset, device=device)
# if __name__ == "__main__":

# PRETRAINING
if __name__ == "__main__" and PRETRAINING:
  config = Config()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # load dataset, transform from folder
  mnist_10_transform = get_transform_MNIST_10(input_size=90)
  trainset, testset = load_MNIST_10(path='./data', transform=mnist_10_transform)
  # embed dataset (3 times 3 patches)
  trainset = Masker(dataset=trainset, config=config).consolidate()
  config.dummy = trainset.__getitem__(0)[0]
  trainset = DataLoader(dataset=trainset, batch_size=config.batch_size)
  # testset = Embedder(dataset=testset, config=config).consolidate()
  # testset = DataLoader(dataset=testset, batch_size=config.batch_size)
  model = ViT(config=config)
  train(model=model, path=TEACH_SAVE_TO, config=config, trainset=trainset, device=device)
# if __name__ == "__main__":
```

### Evaluation
`eval.py` is used to evaluate the trained model on the MNIST-10 dataset. It loads the model and embedder, processes the dataset, and computes the accuracy of the model.

```python
if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  config = Config()
  # load dataset, transform from folder
  cifar_10_transform = get_transform_CIFAR_10(input_size=90)
  trainset, testset = load_CIFAR_10(path='./data', transform=cifar_10_transform)
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
```