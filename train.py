import tinygym as tgym
tgym.set_seed(SEED:=1)

env = tgym.envs.Grid(n_agents=2**14)
learn = tgym.Learner(env=env.reset(SEED))
#learn.model = frl.LSTMPolicy(learn.env, n_hidden=128)
curves = learn.fit(40, steps=16, pbar_desc='reward', lr=1e-16)  # normal lr -> nan (add init_orthogonal)
tgym.print_ascii_curves(curves, keys='loss')
tgym.render_ascii(learn, fps=4)
#tgym.print_table(learn)
env.close()
