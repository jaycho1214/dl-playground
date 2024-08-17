import tqdm
import torch as T
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from model import XOR

data = T.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=T.float32)
label = T.tensor([[0], [1], [1], [0]], dtype=T.float32)

model = XOR()
optim = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
losses = []

model.train()
for _ in tqdm.tqdm(range(10_000)):
    pred = model(data)
    loss = criterion(pred, label)
    losses.append(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()

model.eval()
table = [["Data", "Prediction", "Label"]]
for d, l in zip(data, label):
    table.append([d, model(d).item(), l.item()])

tab = PrettyTable(table[0])
tab.add_rows(table[1:])
print(tab)

plt.plot(losses)
plt.show()