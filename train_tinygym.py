import math
import numpy as np
from tqdm import tqdm
from tinygrad import nn, dtypes, Tensor


class Policy:
    def __init__(self, env, hidden_size=128):
        self.encoder = nn.Linear(math.prod(env.obs.shape[1:]), hidden_size)
        self.actor = nn.Linear(hidden_size, env.n_acts)
        self.value_head = nn.Linear(hidden_size, 1)
        # uncomment block below to init the layers like torch
        # self.encoder.weight = Tensor.kaiming_uniform(self.encoder.weight.shape)
        # self.encoder.bias = Tensor.kaiming_uniform(self.encoder.bias.shape)
        # self.actor.weight = Tensor.kaiming_uniform(self.actor.weight.shape)
        # self.actor.bias = Tensor.kaiming_uniform(self.actor.bias.shape)
        # self.value_head.weight = Tensor.kaiming_uniform(self.value_head.weight.shape)
        # self.value_head.bias = Tensor.kaiming_uniform(self.value_head.bias.shape)

    def __call__(self, x, act=None, with_entropy=None):
        with_entropy = act is not None if with_entropy is None else with_entropy
        h = self.encoder(x.view(x.shape[0], -1)).relu()
        value = self.value_head(h).view(-1)
        x = self.actor(h)
        act = sample_multinomial(x.softmax()) if act is None else act
        x = x - x.logsumexp(axis=-1)[..., None]
        logprob = x.gather(-1, act[..., None]).view(-1)
        entropy = -(x * x.softmax()).sum(-1) if with_entropy else None
        return act.cast(dtypes.uint8), logprob, entropy, value


def sample_multinomial(weights):
    return (weights.cumsum(1) < Tensor.rand_like(weights[:, :1])).sum(1)


class Learner:
    def __init__(self, env, model=None, **kwargs):
        self.env = env
        self.model = Policy(self.env, **kwargs) if model is None else model

    def fit(self, iters=40, steps=16, lr=.01, bs=None, anneal_lr=True, **hparams):
        bs = bs or len(self.env.obs) // 2
        opt = nn.optim.Adam(nn.state.get_parameters(self.model), lr=lr, eps=1e-5)
        pbar = tqdm(range(iters), total=iters)
        curves = []
        for i in pbar:
            opt.lr = lr * (1 - i / iters) if anneal_lr else lr
            obs, values, acts, logprobs, rewards, dones = rollout(self.env, self.model, steps)
            metrics = ppo(self.model, opt, obs, values, acts, logprobs, rewards, dones, bs=bs, **hparams)
            pbar.set_description(f'reward: {rewards.mean().numpy():.3f}')
            if i: pbar.set_postfix_str(f'{1e-6 * acts.numel() * pbar.format_dict["rate"]:.1f}M steps/s')
            curves.append(metrics)
        return {k: [m[k].item() for m in curves] for k in curves[0]}


def rollout(env, model, steps):
    obs, values, acts, logprobs, rewards, dones = [], [], [], [], [], []
    for i in range(steps):
        o = Tensor(env.obs, dtype=dtypes.float32, requires_grad=False)
        with Tensor.test():
            act, logp, _, value = model(o)
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
        value_coef=.5, entropy_coef=.01, norm_adv=True):
    advs = get_advantages(values, rewards, dones, gamma=gamma, gae_lambda=gae_lambda)
    obs, values, acts, logprobs, advs = [xs.view(-1, bs, *xs.shape[2:]) for xs in [obs, values, acts, logprobs, advs]]
    returns = advs + values
    metrics, metric_keys = [], ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'kl']
    for o, old_value, act, old_logp, adv, ret in zip(obs, values, acts, logprobs, advs, returns):
        with Tensor.train():
            _, logp, entropy, value = model(o, act=act)
            logratio = logp - old_logp
            ratio = logratio.exp()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) if norm_adv else adv
            policy_loss = (-adv * ratio).stack(-adv * ratio.clip(1 - clip_coef, 1 + clip_coef)).max(0).mean()
            # usage of .abs().clip(min_=...) below is only needed in tinygrad to enable learning (not needed in torch)
            value_loss = .5 * ((value - ret).abs().clip(min_=1e-8).pow(2)).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
            opt.zero_grad()
            loss.backward()
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


if __name__ == '__main__':
    import tinygym as gym
    gym.set_seed(SEED := 1)
    # To make it faster uses Grid, reduced n_agents and no LSTM (+smaller hidden_size)
    learn = Learner(env=gym.envs.Grid(n_agents=2**13).reset(SEED), hidden_size=64)
    curves = learn.fit(40, steps=16, lr=1e-2, bs=2**13)
    gym.print_curve(curves['loss'], label='loss')
    gym.print_curve(curves['policy_loss'], label='policy_loss')
    gym.print_curve(curves['value_loss'], label='value_loss')
    gym.print_curve(curves['entropy_loss'], label='entropy_loss')
    #gym.play(learn.env, learn.model, fps=4, playable=False)
    learn.env.close()
