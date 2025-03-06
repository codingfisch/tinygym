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


def print_ascii_curves(data_dict, keys=None, height=8, width=65):
    keys = data_dict.keys() if keys is None else (keys,) if isinstance(keys, str) else keys
    for k in keys:
        fig = plotille.Figure()
        fig._height, fig._width = height, width
        fig.y_label = k
        fig.scatter(list(range(len(data_dict[k]))), data_dict[k])
        print('\n'.join(fig.show().split('\n')[:-2]))


def render_ascii(learn, keys=None, fps=4, env_idx=0):
    keys = learn.scalar_data_keys if keys is None else [keys] if isinstance(keys, str) else keys
    obs = (learn._data['obs'] - learn._data['obs'].min()) / (learn._data['obs'].max() - learn._data['obs'].min())
    obs = (23 * obs[env_idx]).numpy().astype(np.uint8) + 232
    for i, o in enumerate(obs):
        print(f'step {i}')
        for row in range(o.shape[0]):
            for col in range(o.shape[1]):
                print(f"\033[48;5;{o[row, col]}m  \033[0m", end='')
            if row < len(keys):
                print(f'{keys[row]}: {learn._data[keys[row]].numpy()[env_idx, i]:.2g}', end='')
            print()
        if i < len(obs) - 1:
            time.sleep(1 / fps)
            print(f'\033[A\033[{len(o) + 1}A')


def print_table(learn, keys=None, fmt='%.2f', env_idx=0):
    keys = learn.scalar_data_keys if keys is None else [keys] if isinstance(keys, str) else keys
    x = np.stack([learn._data[k][env_idx].numpy().astype(np.float32) for k in keys], 1)
    return np.savetxt(fname=sys.stdout.buffer, X=x, fmt=fmt, delimiter='\t', header='\t'.join(keys), comments='')
