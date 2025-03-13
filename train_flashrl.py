import math
import torch
import random
import numpy as np
from tqdm import tqdm
DEVICE = 'cuda'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Policy(torch.nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.encoder = torch.nn.Linear(math.prod(env.obs.shape[1:]), hidden_size)
        self.actor = torch.nn.Linear(hidden_size, env.n_acts)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, act=None, with_entropy=None):
        with_entropy = act is not None if with_entropy is None else with_entropy
        h = self.encoder(x.view(len(x), -1)).relu()
        value = self.value_head(h).view(-1)
        x = self.actor(h)
        act = torch.multinomial(x.softmax(dim=-1), 1)[:, 0] if act is None else act
        x = x - x.logsumexp(dim=-1, keepdim=True)
        logprob = x.gather(-1, act[..., None].long()).view(-1)
        entropy = -(x * x.softmax(dim=-1)).sum(-1) if with_entropy else None
        return act.byte(), logprob, entropy, value


class Learner:
    def __init__(self, env, model=None, **kwargs):
        self.env = env
        self.model = Policy(self.env, **kwargs).to(device=DEVICE) if model is None else model

    def fit(self, iters=40, steps=16, lr=.01, bs=None, anneal_lr=True, **hparams):
        bs = bs or len(self.env.obs) // 2
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        pbar = tqdm(range(iters), total=iters)
        curves = []
        for i in pbar:
            opt.lr = lr * (1 - i / iters) if anneal_lr else lr
            obs, values, acts, logprobs, rewards, dones = rollout(self.env, self.model, steps)
            metrics = ppo(self.model, opt, obs, values, acts, logprobs, rewards, dones, bs=bs, **hparams)
            pbar.set_description(f'reward: {rewards.mean().cpu().numpy():.3f}')
            if i: pbar.set_postfix_str(f'{1e-6 * acts.numel() * pbar.format_dict["rate"]:.1f}M steps/s')
            curves.append(metrics)
        return {k: [m[k].item() for m in curves] for k in curves[0]}


def rollout(env, model, steps):
    obs, values, acts, logprobs, rewards, dones = [], [], [], [], [], []
    for i in range(steps):
        o = torch.from_numpy(env.obs).to(dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            act, logp, _, value = model(o)
        obs.append(o.detach())
        values.append(value.detach())
        acts.append(act.detach())
        logprobs.append(logp.detach())
        rewards.append(env.rewards.copy())
        dones.append(env.dones.copy())
        env.step(act.cpu().numpy())
    return tuple([torch.stack(xs, dim=1) for xs in [obs, values, acts, logprobs]] +
                 [torch.from_numpy(np.stack(xs, axis=1)).to(dtype=torch.float32, device=DEVICE) for xs in [rewards, dones]])


def ppo(model, opt, obs, values, acts, logprobs, rewards, dones, bs=2**13, gamma=.99, gae_lambda=.95,
        clip_coef=.1, value_coef=.5, entropy_coef=.01, norm_adv=True):
    advs = get_advantages(values, rewards, dones, gamma=gamma, gae_lambda=gae_lambda)
    obs, values, acts, logprobs, advs = [xs.view(-1, bs, *xs.shape[2:]) for xs in [obs, values, acts, logprobs, advs]]
    returns = advs + values
    metrics, metric_keys = [], ['loss', 'policy_loss', 'value_loss', 'entropy_loss']
    for o, old_value, act, old_logp, adv, ret in zip(obs, values, acts, logprobs, advs, returns):
        _, logp, entropy, value = model(o, act=act)
        logratio = logp - old_logp
        ratio = logratio.exp()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if norm_adv else adv
        #policy_loss = torch.max(-adv * ratio, -adv * ratio.clip(1 - clip_coef, 1 + clip_coef)).mean()
        policy_loss = (-adv * ratio.clip(1 - clip_coef, 1 + clip_coef)).mean()
        value_loss = .5 * ((value - ret).abs().clip(min=1e-8) ** 2).mean()  # .abs().clip(min_=...) avoids nans in tinygym
        entropy = entropy.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
        metrics.append([loss, policy_loss, value_loss, entropy])
    return {k: torch.stack([values[i] for values in metrics]).mean() for i, k in enumerate(metric_keys)}


def get_advantages(values, rewards, dones, gamma=.99, gae_lambda=.95):  # see arxiv.org/abs/1506.02438 eq. (16)-(18)
    advs = torch.zeros_like(values)
    not_dones = 1. - dones
    for t in range(1, dones.shape[1]):
        delta = rewards[:, -t] + gamma * values[:, -t] * not_dones[:, -t] - values[:, -t-1]
        advs[:, -t-1] = delta + gamma * gae_lambda * not_dones[:, -t] * advs[:, -t]
    return advs


if __name__ == '__main__':
    import tinygym as gym
    set_seed(SEED := 1)
    # To make it faster uses Grid, reduced n_agents and no LSTM (+smaller hidden_size)
    learn = Learner(env=gym.envs.Grid(n_agents=2**13).reset(SEED), hidden_size=64)
    curves = learn.fit(40, steps=16, lr=1e-2, bs=2**13)
    gym.print_curve(curves['loss'], label='loss')
    gym.print_curve(curves['policy_loss'], label='policy_loss')
    gym.print_curve(curves['value_loss'], label='value_loss')
    gym.print_curve(curves['entropy_loss'], label='entropy_loss')
    #gym.play(learn.env, learn.model, fps=4, playable=False)
    learn.env.close()
