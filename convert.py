from pathlib import Path
import numpy as np

from data import load_csv_file

root = Path('C:\data\pipe\Application 1 - Thickness')

answer_file = root / 'answers-384-rows-times-1428-columns.csv'
answers = np.loadtxt(str(answer_file), delimiter=',')

index = 1

data_l = []
answer_l = []

for index in range(1, 385):
    data = load_csv_file(root, index)
    for i, row in enumerate(data):
        data_l.append(data[i])
        answer_l.append(answers[index-1, i])

data = np.stack(data_l, axis=0)
answers = np.stack(answer_l, axis=0)

new_data_file = root / 'data'
new_answer_file = root / 'answer'

np.save(new_data_file, data)
np.save(new_answer_file, answers)
