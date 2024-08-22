<div align="center">
  <p>Aug 18, 2024</p>
  <h1>CIFAR-10</h1>
  <p>
    <img
      src="https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png" 
      style="background: #fff;" 
    />
    <em>Source: CIFAR-10</em>
  </p>
</div>

> The CIFAR-10 dataset consists of 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. Of these, 50,000 are training images and 10,000 are test images.

The CIFAR-10 dataset is widely used as a benchmark for image classification tasks. However, in this project, I will use Generative Adversarial Networks (GANs) to generate new images based on the CIFAR-10 dataset.

# Dataset

I utilize the CIFAR-10 dataset from the built-in `torchvision` library as shown below:

```python
from torchvision import datasets, transforms

training_data = datasets.CIFAR10(
  root="data",
  train=True,
  download=True,
  transform=transforms.Compose([
    transforms.ToTensor(),
  ]),
)
test_data = datasets.CIFAR10(
  root="data",
  train=False,
  download=True,
  transform=transforms.Compose([
    transforms.ToTensor(),
  ]),
)
```

# Generator & Discriminator

## Troubleshooting:

Initially, I experimented with using ResNet as the discriminator model. However, I found that using such a complex architecture for a task as simple as distinguishing between real and fake images was unnecessary.

The generator initially had 3 layers but lacked `self.conv1` and `self.conv2`. After training for 1000 epochs without seeing any significant improvement, I increased the model's capacity by adding more parameters and layers.

Both the activation functions and the network structure were heavily inspired by the architecture described in the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"[^1].

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(latent_dim + 1, 512 * 8 * 8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = out.view(-1, 512, 8, 8)
        out = self.upsample(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out
```

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1),
            nn.LeakyReLU(),
        )
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out
```

The generator model produces an image with 3 channels, corresponding to RGB color images, while the discriminator outputs a single value, which represents the probability of the image being real or generated (fake).

# Optimizer & Loss Function

## Troubleshooting:

Again, the choices for the loss function and optimizer settings were largely inspired by the paper[^1].

I switched the activation function of the generator's output layer from `Tanh` to `Sigmoid` to ensure that the output pixel values range between 0 and 1, which is better suited for generating images.

```python
criterion = nn.BCEWithLogitsLoss()
optim_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optim_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
```

# Training

To enable the generator to produce images corresponding to specific CIFAR-10 classes, I concatenated the label tensor to the noise vector. However, this approach did not yield the desired results in my experiments.

```python
g_losses = []
d_losses = []

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
for epoch in tqdm(range(100)):
  generator.train()
  discriminator.train()
  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    valid = T.ones((images.size(0), 1)).to(device)
    fake = T.zeros((images.size(0), 1)).to(device)

    # TRAINING GENERATOR
    optim_g.zero_grad()
    z = T.randn((images.size(0), 100)).to(device)
    z = T.cat((z, labels.reshape((labels.size(0), 1)) / 10), dim=1).to(device)
    generated_imgs = generator(z)
    l = discriminator(generated_imgs)
    g_loss = criterion(l, valid)
    g_losses.append(g_loss.item())
    g_loss.backward()
    optim_g.step()

    # TRAINING DISCRIMINATOR
    optim_d.zero_grad()
    real_loss = criterion(discriminator(images), valid)
    fake_loss = criterion(discriminator(generated_imgs.detach()), fake)
    # Average the losses as recommended in the original GAN paper
    d_loss = (real_loss + fake_loss) / 2
    d_losses.append(d_loss.item())
    d_loss.backward()
    optim_d.step()

  if (epoch + 1) % 10 == 0:
    # Placeholder for additional operations during training every 10 epochs
    pass
```

Although GANs have become integral to many applications today, this was my first time building and training a GAN from scratch. Unlike more straightforward neural networks, training GANs presented significant challenges, and I found the need for more computational power to be substantial.

# Result

The following images showcase the progress of the GAN's image generation at different stages of training. From left to right, the images correspond to the output generated at Epoch 10, Epoch 100, Epoch 200, and Epoch 300.

| **Epoch 10**                                                                                                | **Epoch 100**                                                                                                |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| ![Epoch: 10](/cifar10-gan/images/10.png) | ![Epoch: 100](/cifar10-gan/images/100.png) |

| **Epoch 200**                                                                                                | **Epoch 300**                                                                                                |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| ![Epoch: 200](/cifar10-gan/images/200.png) | ![Epoch: 300](/cifar10-gan/images/300.png) |

[^1]: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
