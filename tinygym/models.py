import math
import numpy as np
from tinygrad.nn import state, dtypes, Tensor, Linear, LSTMCell


class Policy:
    def __init__(self, env, hidden_size=128, lstm=False):
        self.encoder = Linear(math.prod(env.obs.shape[1:]), hidden_size)
        self.actor = Linear(hidden_size, env.n_acts)
        self.value_head = Linear(hidden_size, 1)
        self.lstm = cleanrl_init(LSTMCell(hidden_size, hidden_size)) if lstm else None
        # uncomment block below to init the layers like torch
        # self.encoder.weight = Tensor.kaiming_uniform(self.encoder.weight.shape)
        # self.encoder.bias = Tensor.kaiming_uniform(self.encoder.bias.shape)
        # self.actor.weight = Tensor.kaiming_uniform(self.actor.weight.shape)
        # self.actor.bias = Tensor.kaiming_uniform(self.actor.bias.shape)
        # self.value_head.weight = Tensor.kaiming_uniform(self.value_head.weight.shape)
        # self.value_head.bias = Tensor.kaiming_uniform(self.value_head.bias.shape)

    def __call__(self, x, state=None, act=None, with_entropy=None):
        with_entropy = act is not None if with_entropy is None else with_entropy
        h = self.encoder(x.view(x.shape[0], -1)).relu()
        h, c = (h, None) if self.lstm is None else self.lstm(h, state)
        value = self.value_head(h).view(-1)
        x = self.actor(h)
        act = sample_multinomial(x.softmax()) if act is None else act
        x = x - x.logsumexp(axis=-1)[..., None]
        logprob = x.gather(-1, act[..., None]).view(-1)
        entropy = -(x * x.softmax()).sum(-1) if with_entropy else None
        return act.cast(dtypes.uint8), logprob, entropy, value, (h, c)


def cleanrl_init(module):
    for name, param in state.get_state_dict(module).items():
        if 'bias' in name: param *= 0
        elif 'weight' in name: orthogonal(param)
    return module


def orthogonal(x, gain=1.):
    q, r = np.linalg.qr(x.numpy())
    return Tensor(gain * q * np.sign(np.diag(r, 0)))


def sample_multinomial(weights):
    return (weights.cumsum(1) < Tensor.rand_like(weights[:, :1])).sum(1)
