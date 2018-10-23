import matplotlib.pyplot as plt
import data
import numpy as np
import random
import torch
from pathlib import Path
from model import SimpleNet
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = 'C:\data\pipe\Application 1 - Thickness'
batch_size = 63058 // 12

rundir = Path('runs/run6/')

dataset = data.NonZeroSubset(data.PipeThicknessData())
rnd = torch.load(str(rundir / 'random'))
train, test, rnd = data.random_split(dataset, rnd=rnd)
dataset = test

#model = torch.load('runs/run4/27.mdl').eval().to('cuda')
model = torch.load('runs/run6/15.mdl').eval().to('cuda')

full_loader = DataLoader(test, batch_size=batch_size, drop_last=False)


error = []
batch = tqdm(full_loader, total=len(dataset) // batch_size)
for signal, target_label in batch:
    signal, target_label = signal.to(device), target_label.to(device)
    estimate = model(signal.unsqueeze(1))
    err = estimate - target_label
    error.append(err.data.cpu().numpy())

plt.hist(error)
plt.show()

