import tinygym as tgym
from tinygym import print_table
tgym.set_seed(SEED:=1)

env = tgym.envs.Grid(n_agents=2**10)
learn = tgym.Learner(env=env.reset(SEED))
#learn.model = tgym.LSTMPolicy(learn.env, n_hidden=64)
curves = learn.fit(10, steps=16, lr=1e-25)
print(curves)
tgym.print_ascii_curve(curves['loss'], label='loss')
tgym.print_ascii_curve(curves['policy_loss'], label='policy_loss')
tgym.print_ascii_curve(curves['value_loss'], label='value_loss')
tgym.print_ascii_curve(curves['entropy_loss'], label='entropy_loss')
obs, values, acts, logprobs, rewards, dones = tgym.rollout(learn.env, learn.model, steps=16)
data = {'value': values, 'act': acts, 'logprob': logprobs, 'reward': rewards, 'done': dones}
data = {k: v[0].numpy() for k, v in data.items()}
tgym.render_ascii(obs[0], fps=4, data=data)
print_table(data)
env.close()
