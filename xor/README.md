<div align="center">
  <p>Aug 16, 2024</p>
  <h1>XOR</h1>
  <p>
    <img
      src="/xor/images/plot.png" 
      style="background: #fff;" 
    />
    <em>Source: Me</em>
  </p>
</div>

> XOR (eXclusive OR) compares two input bits. If the bits are the same, the output is 0. If the bits are different, the output is 1. This logic makes XOR a classic non-linearly separable problem.

In neural networks, a **single-layer perceptron** can only learn a decision boundary that is a straight line. However, XOR is **not linearly separable**, meaning it cannot be solved by a single-layer perceptron. This limitation arises because there is no way to draw a straight line to separate the input pairs that map to `1` from those that map to `0` in XOR. Therefore, to solve this problem, we require a more complex architecture, such as a **multi-layer perceptron (MLP)**, which introduces non-linearities and can handle problems where the decision boundaries are not simple straight lines.

Here, we will solve the XOR problem using a neural network with one input layer, one hidden layer, and one output layer. We will experiment with different learning rates and activation functions to observe their effects on performance.

# Prepare Dataset

We can define the XOR data and its corresponding labels as follows:

```python
import torch as T

data = T.tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
], dtype=T.float32)

label = T.tensor([
  [0],
  [1],
  [1],
  [0]
], dtype=T.float32)
```

The dataset consists of four 2-dimensional input vectors corresponding to the four possible combinations of two binary values, and the labels represent the expected XOR output.

# Loss Function

Since XOR is a binary classification problem, we use **Binary Cross Entropy (BCE)** as the loss function. BCE is suitable for binary classification tasks as it quantifies the difference between the predicted probability (from the Sigmoid function) and the actual label, providing a measure to optimize.

```python
import torch.nn as nn

criterion = nn.BCELoss()
```

# Model

For the model architecture, we use a hidden layer with a **LeakyReLU** activation function and an output layer with a **Sigmoid** activation function. **LeakyReLU** is chosen because it avoids the "dying ReLU" problem by allowing a small, non-zero gradient when the unit is not active. **Sigmoid** is used in the final layer since it outputs values between 0 and 1, which is required by the BCE loss function.

```python
class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 3),  # Input layer to hidden layer with 3 neurons
            nn.LeakyReLU(),   # Activation function for hidden layer
            nn.Linear(3, 1),  # Hidden layer to output layer
            nn.Sigmoid()      # Activation function for output layer
        )

    def forward(self, x):
        x = self.layer(x)
        return x

model = XOR()
```

This model takes two input values, transforms them through a hidden layer of three neurons, and outputs a single value representing the XOR result.

# Optimizer

Various optimization algorithms have been proposed in the deep learning literature. For now, we will use **Adam**[^1], which is an efficient and widely-used optimizer due to its adaptive learning rate capabilities. We set the learning rate to $10^{-3}$, but this can be adjusted later if needed.

```python
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

# Training

We train the model for 10,000 epochs. Given the simplicity of the dataset (only 4 samples), we can expect the model to converge quickly. However, with such a small dataset, it is hard to assess generalization performance or overfitting. For more complex tasks, cross-validation or additional regularization techniques would be necessary to ensure the model generalizes well.

Here is the training loop:

```python
import tqdm

model.train()
losses = []
for _ in tqdm.tqdm(range(10_000)):
    pred = model(data)            # Forward pass
    loss = criterion(pred, label) # Compute loss
    losses.append(loss.item())    # Store loss for visualization

    optimizer.zero_grad()         # Zero out gradients
    loss.backward()               # Backward pass (compute gradients)
    optimizer.step()              # Update weights
```

Below is a plot of the loss over time:

<img src="/xor/images/losses.png" />

# Predictions

After training, our model successfully predicts the XOR outputs with high accuracy. Here are the predictions and the corresponding true labels:

```python
+------------------+------------------------+-------+
|       Data       |       Prediction       | Label |
+------------------+------------------------+-------+
| tensor([0., 0.]) | 3.807698885793798e-05  |  0.0  |
| tensor([0., 1.]) |   0.999966025352478    |  1.0  |
| tensor([1., 0.]) |   0.999976396560669    |  1.0  |
| tensor([1., 1.]) | 1.6093748854473233e-05 |  0.0  |
+------------------+------------------------+-------+
```

As seen from the table above, the model has learned the XOR function effectively. The predictions are very close to the expected outputs, demonstrating that the model has successfully captured the XOR logic.

[^1]: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
