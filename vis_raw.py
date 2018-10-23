import matplotlib.pyplot as plt
import data
import numpy as np
import random
import torch
from pathlib import Path
from model import SimpleNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = 'C:\data\pipe\Application 1 - Thickness'

rundir = Path('runs/run6/')

dataset = data.NonZeroSubset(data.PipeThicknessData())
#dataset = data.TransformedDataset(dataset, transform_signal=data.RandomPhaseShift())
rnd = torch.load(str(rundir / 'random'))
train, test, rnd = data.random_split(dataset, rnd=rnd)
dataset = test

#model = torch.load('runs/run4/27.mdl').eval().to('cuda')
model = torch.load('runs/run6/15.mdl').eval().to('cuda')

while True:
    index = random.randrange(0, len(dataset)-1, 1)
    measurement = dataset[index]

    t = np.linspace(0, 511, 512)
    y = measurement[0].numpy()

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    line1 = ax1.plot(t, y)
    estimate = model(measurement[0].unsqueeze(0).unsqueeze(1).to('cuda'))
    ax1.set_title('gt: ' + str(measurement[1].item()) + ' est: ' + str(estimate.item()))


    NFFT = 20  # the length of the windowing segments
    Fs = 20  # the sampling frequency

    Pxx, freqs, bins, im = ax2.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=15)
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot
    plt.show()

    fig.canvas.draw_idle()
    plt.show()