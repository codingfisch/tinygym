import math
from tinygrad.nn import dtypes, Tensor, Linear, LSTMCell
from .qr import QR


class LSTMPolicy:
    def __init__(self, env, n_hidden=128):
        self.encoder = Linear(math.prod(env.obs.shape[1:]), n_hidden)
        self.decoder = Linear(n_hidden, env.n_acts)
        self.value_head = Linear(n_hidden, 1)
        self.lstm = LSTMCell(n_hidden, n_hidden)
        self.lstm.bias_hh *= 0
        self.lstm.bias_ih *= 0
        qr = QR()
        self.lstm.weight_hh = qr(self.lstm.weight_hh)[0]
        self.lstm.weight_ih = qr(self.lstm.weight_ih)[0]

    def __call__(self, x, state=None, act=None, with_entropy=None):
        with_entropy = act is not None if with_entropy is None else with_entropy
        x = self.encoder(x.view(x.shape[0], -1)).relu()
        h, c = self.lstm(x, state)
        value = self.value_head(h)[:, 0]
        x = self.decoder(h)
        act = sample_multinomial(x.softmax()).cast(dtypes.uint8).squeeze() if act is None else act
        x = x - x.logsumexp(axis=-1)[..., None]
        logprob = x.gather(-1, act[..., None])[..., 0]
        entropy = -(x * x.softmax()).sum(-1) if with_entropy else None
        return act, logprob, entropy, value, (h, c)


def sample_multinomial(weights):
    rand_vals = Tensor.rand_like(weights[:, :1])
    return (weights.cumsum(1) > rand_vals).sum(1) - 1
