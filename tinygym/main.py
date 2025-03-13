import numpy as np
from tqdm import tqdm
from tinygrad import nn, dtypes, Tensor
from torch.utils.tensorboard import SummaryWriter

from .models import Policy


class Learner:
    def __init__(self, env, model=None, **kwargs):
        self.env = env
        self.model = Policy(self.env, **kwargs) if model is None else model

    def fit(self, iters=40, steps=16, lr=.01, bs=None, anneal_lr=True, log=False, desc=None, stop_func=None, **hparams):
        bs = bs or len(self.env.obs) // 2
        logger = SummaryWriter() if log else None
        opt = nn.optim.Adam(nn.state.get_parameters(self.model), lr=lr, eps=1e-5)
        pbar = tqdm(range(iters), total=iters)
        curves = []
        for i in pbar:
            opt.lr = lr * (1 - i / iters) if anneal_lr else lr
            obs, values, acts, logprobs, rewards, dones = rollout(self.env, self.model, steps)
            losses = ppo(self.model, opt, obs, values, acts, logprobs, rewards, dones, bs=bs, **hparams)
            if desc:
                desc_v = dones.mean() if desc == 'done' else rewards.mean() if desc == 'reward' else losses[desc]
                pbar.set_description(f'{desc}: {desc_v.numpy():.3f}')
            if i: pbar.set_postfix_str(f'{1e-6 * acts.numel() * pbar.format_dict["rate"]:.1f}M steps/s')
            if log:
                for k, v in losses.items(): logger.add_scalar(k, v, global_step=i)
                for k, v in nn.state.get_state_dict(self.model).items(): logger.add_histogram(k, v.numpy(), i)
            curves.append(losses)
            if stop_func is not None:
                if stop_func(obs, values, acts, logprobs, rewards, dones, **losses): break
        return {k: [m[k].item() for m in curves] for k in curves[0]}


def rollout(env, model, steps, state=None):
    obs, values, acts, logprobs, rewards, dones = [], [], [], [], [], []
    for i in range(steps):
        o = Tensor(env.obs, dtype=dtypes.float32, requires_grad=False)
        with Tensor.test():
            act, logp, _, value, state = model(o, state=state)
        obs.append(o)
        values.append(value.detach())
        acts.append(act)
        logprobs.append(logp)
        rewards.append(env.rewards.copy())
        dones.append(env.dones.copy())
        env.step(act.numpy())
    obs = Tensor.stack(*obs, dim=1)
    values = Tensor.stack(*values, dim=1)
    acts = Tensor.stack(*acts, dim=1)
    logprobs = Tensor.stack(*logprobs, dim=1)
    rewards = Tensor(np.stack(rewards, axis=1).astype(np.float32))
    dones = Tensor(np.stack(dones, axis=1).astype(np.float32))
    return obs, values, acts, logprobs, rewards, dones


def ppo(model, opt, obs, values, acts, logprobs, rewards, dones, bs=2**13, gamma=.99, gae_lambda=.95, clip_coef=.1,
        value_coef=.5, value_clip_coef=.1, entropy_coef=.01, max_grad_norm=.5, norm_adv=True, state=None):
    advs = get_advantages(values, rewards, dones, gamma=gamma, gae_lambda=gae_lambda)
    obs, values, acts, logprobs, advs = [xs.view(-1, bs, *xs.shape[2:]) for xs in [obs, values, acts, logprobs, advs]]
    returns = advs + values
    metrics, metric_keys = [], ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'kl']
    for o, old_value, act, old_logp, adv, ret in zip(obs, values, acts, logprobs, advs, returns):
        with Tensor.train():
            _, logp, entropy, value, state = model(o, state=state, act=act)
            state = state if model.lstm is None else (state[0].detach(), state[1].detach())
            logratio = logp - old_logp
            ratio = logratio.exp()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) if norm_adv else adv
            policy_loss = (-adv * ratio).stack(-adv * ratio.clip(1 - clip_coef, 1 + clip_coef)).max(0).mean()
            # usage of .abs().clip(min_=...) below is only needed in tinygrad to enable learning (not needed in torch)
            if value_clip_coef is None:
                value_loss = .5 * ((value - ret).abs().clip(min_=1e-8).pow(2)).mean()
            else:
                v_clipped = old_value + (value - old_value).clip(-value_clip_coef, value_clip_coef)
                value_loss = .5 * (value - ret).abs().clip(min_=1e-8).pow(2).stack((v_clipped - ret).abs().clip(min_=1e-8).pow(2)).max(0).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
            opt.zero_grad()
            loss.backward()
            if max_grad_norm is not None: clip_grad_norm(opt.params, max_grad_norm)
            opt.step()
        kl = ((ratio - 1) - logratio).mean()
        metrics.append([loss, policy_loss, value_loss, entropy_loss, kl])
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
