import torchvision as tv


def get_transform_MNIST(input_size=90):
  return tv.transforms.Compose([
    # 1. Augmentation for better generalization
    tv.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(brightness=0.1, contrast=0.1),  # If RGB
    # 2. Resize and ToTensor
    tv.transforms.Resize((input_size, input_size)),
    tv.transforms.ToTensor(),
    # 3. Normalization using ImageNet statistics for pre-trained models
    tv.transforms.Normalize(
      mean=[0.485],
      std=[0.229]
    ),
  ])  # TRANSFORM
# get_transform_MNIST_10