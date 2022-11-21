import torch
import torchvision

batch_size = 100

# 28 * 28 gray scale images
# 60 000 training image
# 10 000 test image
train_dataset = torchvision.datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root = './data',
    train = False,
    transform = torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True
        )

test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = True
)

"""for i,(features,label) in enumerate(train_loader):
    print(features.shape) # [100,1,28,28] 
    print(label.shape) # [100]"""
    