import sys
import time
import random
import plotille
import numpy as np
from tinygrad import Tensor


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    Tensor.manual_seed(seed)


def print_ascii_curve(array, label=None, height=8, width=65):
    fig = plotille.Figure()
    fig._height, fig._width = height, width
    fig.y_label = fig.y_label if label is None else label
    fig.scatter(list(range(len(array))), array)
    print('\n'.join(fig.show().split('\n')[:-2]))


def render_ascii(obs, fps=4, data=None):
    obs = (23 * (obs - obs.min()) / (obs.max() - obs.min())).numpy().astype(np.uint8) + 232
    for i, o in enumerate(obs):
        print(f'step {i}')
        for row in range(o.shape[0]):
            for col in range(o.shape[1]):
                print(f"\033[48;5;{o[row, col]}m  \033[0m", end='')
            if row < len(data):
                print(f'{list(data.keys())[row]}: {list(data.values())[row][i]:.2g}', end='')
            print()
        if i < len(obs) - 1:
            time.sleep(1 / fps)
            print(f'\033[A\033[{len(o) + 1}A')


def print_table(data, fmt='%.2f'):
    x = np.stack(list(data.values()), 1)
    np.savetxt(fname=sys.stdout.buffer, X=x, fmt=fmt, delimiter='\t', header='\t'.join(data.keys()), comments='')
