from unittest import TestCase
import data
from torch.utils.data import random_split, ConcatDataset, DataLoader


class TestDataLoaders(TestCase):
    def test_dataloaders(self):
        batch_size = 123058 // 12

        epochs = 200

        dataset = data.NonZeroSubset(data.PipeThicknessData())

        train_length = len(dataset) * 9 // 10
        test_length = len(dataset) - train_length

        train, test = random_split(dataset, [train_length, test_length])

        phase_shifted = data.TransformedDataset(train, transform_signal=data.RandomPhaseShift())
        train = ConcatDataset([train, phase_shifted])

        train_loader = DataLoader(phase_shifted, batch_size=batch_size, shuffle=True, drop_last=False)

        for minibatch in train_loader:
            pass