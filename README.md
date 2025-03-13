# tinygym
`tinygym` reimplements [`flashrl`](https://github.com/codingfisch/flashrl) while using [`tinygrad`](https://github.com/tinygrad/tinygrad) instead of [`torch`](https://github.com/pytorch/pytorch)

🛠️ `pip install tinygym` or clone the repo & `pip install -r requirements.txt`
  - If cloned (or if envs changed), compile: `python setup.py build_ext --inplace`

The [README of `flashrl`](https://github.com/codingfisch/flashrl) is mostly valid for `tinygym` with the biggest difference being:
  - **`tinygym` is not fast (yet)** -> Learns Pong in ~5 minutes instead of 5 seconds (on a RTX 3090)

Just like in `flashrl`, `python train.py` should look like this (with the progress bar moving ~60x slower):
<p align="center">
  <img src="https://github.com/user-attachments/assets/62da23a8-4d30-41f8-8843-1267e43a8744">
</p>

Check out the onefile branch, if you want to make it fast(=try to make TinyJit work)!

# Implementation differences to `flashrl`
The **most important difference** (enabled RL after 2 hours of debugging):
- **Use `.abs().clip(min_=1e-8)` in `ppo` to avoid close to zero values in `(value - ret)`**

Without this the optimizer step can result in NaNs and **"RL doesn't work"** 😜

To potentially enable `tinygrad.TinyJit` (does not work yet, hence the slowness)
- `Learner` does not `.setup_data` and
- `rollout` is a function (instead of a `Learner`-method) that fills a list with Tensors and `.stack`s them at the end

Since it somehow performs better
- `.uniform` (`tinygrad` default) instead of `.kaiming_uniform` (`torch` default) weight initialization for `nn.Linear`

Custom `tinygrad`-rewrites of `torch.nn.init.orthogonal_` & `torch.nn.utils.clip_grad_norm_`are used

You'll find a `.detach()` here and a `.contiguous()` but other than that `tinygym`=`flashrl` 🤝

## Acknowledgements 🙌
I want to thank
- [George Hotz](https://github.com/geohot) and the tinygrad team for commoditizing the petaflop! Star [tinygrad](https://github.com/tinygrad/tinygrad) ⭐
- [Andrej Karpathy](https://github.com/karpathy) for commoditizing RL knowledge! Star [pg-pong](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) ⭐

and last but not least...

<p align="center">
  <img src="https://media1.tenor.com/m/ibYVxrR2hOgAAAAC/well-done.gif">
</p>
