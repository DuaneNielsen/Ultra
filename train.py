import torch
import torchvision.models as models
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
import data
from model import Net, SimpleNet
from pathlib import Path
import statistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_id = 'run6'
batch_size = 123058 // 12
epochs = 200

basepath = Path('runs/' + run_id)
basepath.mkdir(parents=True, exist_ok=True)

dataset = data.NonZeroSubset(data.PipeThicknessData())

train_length = len(dataset) * 9//10
test_length = len(dataset) - train_length

train, test, random = data.random_split(dataset)
torch.save(random, str(basepath / 'random'))

augmented = [train]
for _ in range(7):
    phase_shifted = data.TransformedDataset(train, transform_signal=data.RandomPhaseShift())
    augmented.append(phase_shifted)

train = ConcatDataset(augmented)


train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False)

model = SimpleNet().to(device)
optim = Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()


for epoch in range(epochs):
    train_batch = tqdm(train_loader, total=len(train) / batch_size, unit='batches')
    for signal, target_label in train_batch:
        signal, target_label = signal.to(device), target_label.to(device)

        model = model.train()
        optim.zero_grad()
        label = model(signal.unsqueeze(1))
        loss = criterion(label, target_label)
        loss.backward()
        train_batch.set_description('Train Loss: %4f' % loss.item())
        optim.step()


    test_losses = []
    test_batch = tqdm(test_loader, total=len(test) // batch_size)
    for signal, target_label in test_batch:
        signal, target_label = signal.to(device), target_label.to(device)

        model = model.eval()
        label = model(signal.unsqueeze(1))
        loss = criterion(label, target_label)
        test_losses.append(loss.item())
        test_batch.set_description('Epoch %i Test Loss: %4f' % (epoch, statistics.mean(test_losses)))

    model_file = basepath / Path(str(epoch) + '.mdl')
    torch.save(model, str(model_file))