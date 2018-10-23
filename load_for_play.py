from pathlib import Path
import numpy as np

path = Path('C:\data\pipe\Application 1 - Thickness')
datafile = path / 'data.npy'
data = np.load(datafile)
answerfile = path / 'answer.npy'
answers = np.load(answerfile)