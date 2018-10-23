import torch
import matplotlib.pyplot as plt
import numpy as np

from data import get_datasets
import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('runs/run6/15.mdl').eval().to(device)

root = Path('C:\data\pipe\Application 1 - Thickness')

batch_size = 63058 // 8
epochs = 40

dataset, dev, train, test = get_datasets()

full_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                          pin_memory=torch.cuda.is_available())

# train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False,
#                           pin_memory=torch.cuda.is_available())
#
# test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False,
#                          pin_memory=torch.cuda.is_available())


estimate_l = []
batch = tqdm(full_loader, total=len(dataset) // batch_size)
for signal, target_label in batch:
    signal, target_label = signal.to(device), target_label.to(device)
    estimate = model(signal.unsqueeze(1))
    estimate_l.append(estimate.data.cpu().numpy())

estimate = np.concatenate(estimate_l)

y_axis = estimate.shape[0] // 383
estimate = estimate.reshape(383, y_axis)

answers = data.get_answers(root)

fig, (ax0, ax1) = plt.subplots(2, 1)

c = ax0.pcolor(answers)
ax0.set_title('ground truth')

c = ax1.pcolor(estimate)
ax1.set_title('estimate')

fig.tight_layout()
plt.show()