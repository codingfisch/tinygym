import tinygym as gym
gym.set_seed(SEED:=1)

env = gym.envs.Pong(n_agents=2**14).reset(SEED)  # try one of: Pong, Grid, MultiGrid
learn = gym.Learner(env, hidden_size=128, lstm=True)
curves = learn.fit(40, steps=16, lr=1e-2, desc='done')
gym.print_curve(curves['loss'], label='loss')
gym.play(env, learn.model, fps=8, playable=False)  # if env is MultiGrid, try obs='total_obs', with_total_obs=True
env.close()
