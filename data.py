import torch.utils.data
from pathlib import Path
import numpy as np
from torch.utils.data import Subset
import random


class PipeThicknessData(torch.utils.data.Dataset):
    def __init__(self):
        torch.utils.data.Dataset.__init__(self)
        path = Path('C:\data\pipe\Application 1 - Thickness')
        datafile = path / 'data.npy'
        self.data = np.load(datafile)
        answerfile = path / 'answer.npy'
        self.answers = np.load(answerfile)

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.Tensor([self.answers[index]])

    def __len__(self):
        return self.data.shape[0]


class NonZeroSubset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.indices = dataset.answers.nonzero()[0]
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.indices = indices
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)


def random_split(dataset, rnd=None):
    if rnd is None:
        rnd = torch.rand(len(dataset))
    mask = rnd <= 0.8
    indices = mask.nonzero().squeeze()
    train = IndexedSubset(dataset, indices)
    mask = rnd > 0.8
    indices = mask.nonzero().squeeze()
    test = IndexedSubset(dataset, indices)
    return train, test, rnd


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_signal=None, transform_target=None):
        self.dataset = dataset
        self.transform_signal = transform_signal
        self.transform_target = transform_target

    def __getitem__(self, x):
        signal, target = self.dataset[x]
        if self.transform_signal is not None:
            signal = self.transform_signal(signal)
        if self.transform_target is not None:
            target = self.transform_target(target)
        return signal, target

    def __len__(self):
        return len(self.dataset)


class RandomPhaseShift:
    def __call__(self, signal):
        phase_shift = random.randrange(-20, 20)
        if phase_shift >= 0:
            shifted_signal = torch.cat((torch.zeros(phase_shift), signal[phase_shift:]))
        else:
            shifted_signal = torch.cat((signal[:phase_shift], torch.zeros(-phase_shift)))
        return shifted_signal


def get_answers(root):
    root_path = Path(root)
    answer_file = root_path / 'answers-384-rows-times-1428-columns.csv'
    answers = np.loadtxt(str(answer_file), delimiter=',')
    return answers


def get_datasets():
    dataset = PipeThicknessData()
    dev = Subset(dataset, range(len(dataset) * 2 // 10))
    train = Subset(dataset, range(0, len(dataset) * 9 // 10))
    test = Subset(dataset, range(len(dataset) * 9 // 10 + 1, len(dataset)))
    return dataset, dev, train, test


def load_csv_file(root, index):
    root_path = Path(root)
    file = root_path / 'one-csv-data-file-for-each-row-in-answers' / ('row-%04d' % index)
    file = file.with_suffix('.csv')
    data = np.loadtxt(str(file), delimiter=',')
    return data

