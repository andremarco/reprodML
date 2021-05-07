import torch
import torchaudio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as Path
import os

torch.cuda.is_available()
print(os.getcwd())

csv = pd.read_csv('./data/ESC-50-master/meta/esc50.csv')
x, sr = torchaudio.load(f'./data/ESC-50-master/audio/{csv.iloc[0, 0]}')
h = torchaudio.transforms.Resample(new_freq=8000)(x)
print(h.shape)
plt.imshow(h[0])
