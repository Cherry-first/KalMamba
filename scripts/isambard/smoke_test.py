"""Check the image: GPU, the compiled Mamba kernels and headless MuJoCo rendering."""
import torch

assert torch.cuda.is_available(), "no GPU visible"
print("torch", torch.__version__, "on", torch.cuda.get_device_name(0))

from mamba_ssm import Mamba

# Settings from Table 2 of the KalMamba paper; d_conv=2 is below the library default of 4.
mamba = Mamba(d_model=256, d_state=64, d_conv=2, expand=2).cuda()
x = torch.randn(4, 256, 256, device="cuda", requires_grad=True)
y = mamba(x)
y.sum().backward()
assert y.shape == x.shape and torch.isfinite(x.grad).all()
print("mamba forward/backward OK")

from dm_control import suite

env = suite.load("walker", "walk", task_kwargs={"random": 0})
env.reset()
img = env.physics.render(height=64, width=64, camera_id=0)
assert img.shape == (64, 64, 3) and img.std() > 0, "blank render"
print("dm_control EGL render OK", img.shape)

import gym  # noqa: F401  (used by the VRKN environment wrappers)

print("SMOKE TEST PASSED")
