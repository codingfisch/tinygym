from tqdm import tqdm
from tinygrad import TinyJit
from tinygrad import nn, dtypes, Tensor

from .models import LSTMPolicy


class Learner:
    def __init__(self, env, model=None):
        self.env = env
        self.model = LSTMPolicy(self.env) if model is None else model
        self._data, self._np_data = None, None

    @property
    def scalar_data_keys(self):
        return [k for k, v in self._data.items() if v.ndim == 2]

    def fit(self, iters=40, steps=16, lr=.01, anneal_lr=True, pbar_desc=None, target_kl=None, **hparams):
        self.setup_data(steps)
        curves = []
        opt = nn.optim.Adam(nn.state.get_parameters(self.model), lr=lr, eps=1e-5)
        pbar = tqdm(range(iters), total=iters)
        for i in pbar:
            opt.lr = lr * (1 - i / iters) if anneal_lr else lr
            self.rollout(steps)
            metrics = ppo(self.model, opt, **self._data, **hparams)
            pbar.set_description(f'{pbar_desc}: {self._data[pbar_desc + "s"].numpy().mean():.3f}')
            if i: pbar.set_postfix_str(f'{1e-6 * self._data["acts"].numel() * pbar.format_dict["rate"]:.1f}M steps/s')
            curves.append(metrics)
            if target_kl is not None:
                if metrics['kl'] > target_kl: break
        return {k: [m[k].item() for m in curves] for k in curves[0]}

    def setup_data(self, steps):
        values = Tensor.empty((len(self.env.obs), steps), requires_grad=False)
        obs = Tensor.empty((*values.shape, *self.env.obs.shape[1:]), requires_grad=False)
        self._data = {'obs': obs, 'values': values, 'acts': values.clone().cast(dtypes.uint8), 'logprobs': values.clone()}
        self._np_data = {'rewards': values.cast(dtypes.int8).numpy(), 'dones': values.cast(dtypes.int8).numpy()}

    @TinyJit
    def rollout(self, duration, state=None):
        for i in range(duration):
            o = Tensor(self.env.obs)#, requires_grad=False)
            act, logp, _, value, state = self.model(o, state=state)
            value.requires_grad = False
            act.requires_grad = False
            logp.requires_grad = False
            self._data['obs'][:, i] = o
            self._data['values'][:, i] = value
            self._data['acts'][:, i] = act
            self._data['logprobs'][:, i] = logp
            self._np_data['rewards'][:, i] = self.env.rewards
            self._np_data['dones'][:, i] = self.env.dones
            self.env.step(act.numpy())
        self._data.update({k: Tensor(v) for k, v in self._np_data.items()})

@TinyJit
def ppo(model, opt, obs, values, acts, logprobs, rewards, dones, bs=8192, gamma=.99, gae_lambda=.95, clip_coef=.1,
        value_coef=.5, value_clip_coef=.1, entropy_coef=.01, max_grad_norm=.5, norm_adv=True, state=None):
    Tensor.training = True
    advs = get_advantages(values, rewards, dones, gamma=gamma, gae_lambda=gae_lambda)
    obs, values, acts, logprobs, advs = [xs.view(-1, bs, *xs.shape[2:]) for xs in [obs, values, acts, logprobs, advs]]
    returns = advs + values
    metrics, metric_keys = [], ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'kl']
    for o, old_value, act, old_logp, adv, ret in zip(obs, values, acts, logprobs, advs, returns):
        _, logp, entropy, value, state = model(o, state=state, act=act)
        state = (state[0].detach(), state[1].detach())
        logratio = logp - old_logp
        ratio = logratio.exp()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if norm_adv else adv
        policy_loss = (-adv * ratio).stack(-adv * ratio.clip(1 - clip_coef, 1 + clip_coef)).max(0).mean()
        if value_clip_coef:
            v_clipped = old_value + (value - old_value).clip(-value_clip_coef, value_clip_coef)
            value_loss = .5 * (value - ret).pow(2).stack((v_clipped - ret).pow(2)).max(0).mean()
        else:
            value_loss = .5 * (value - ret).pow(2).mean()
        entropy = entropy.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(opt.params, max_grad_norm)
        opt.step()
        kl = ((ratio - 1) - logratio).mean()
        metrics.append([loss, policy_loss, value_loss, entropy, kl])
    return {k: Tensor.stack(*[values[i] for values in metrics]).mean() for i, k in enumerate(metric_keys)}


def get_advantages(values, rewards, dones, gamma=.99, gae_lambda=.95):  # see arxiv.org/abs/1506.02438 eq. (16)-(18)
    advs = Tensor.zeros_like(values).contiguous()
    not_dones = 1. - dones
    for t in range(1, dones.shape[1]):
        delta = rewards[:, -t] + gamma * values[:, -t] * not_dones[:, -t] - values[:, -t-1]
        advs[:, -t-1] = delta + gamma * gae_lambda * not_dones[:, -t] * advs[:, -t]
    return advs


def clip_grad_norm(parameters, max_norm, norm_type=2):
    total_norm = Tensor.cat(*[p.grad.view(-1) for p in parameters]).pow(norm_type).pow(1 / norm_type).sum()
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = clip_coef.clip(min_=1)
    for p in parameters:
        p.grad *= clip_coef
