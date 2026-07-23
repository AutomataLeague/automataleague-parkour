"""Validate the full ParkourEnvWarp (torch + torchrl + Warp) on GPU + benchmark."""

import time

import torch

from automataleague.envs.parkour.parkour_warp import ParkourEnvWarp

env = ParkourEnvWarp(robot="spot", num_envs=64, device="cuda")
td = env.reset()
print(f"reset: observation {tuple(td['observation'].shape)}", flush=True)

td["action"] = torch.zeros(64, env.robot.n_joints, device="cuda")
td = env.step(td)
finite = bool(torch.isfinite(td["next", "reward"]).all())
print(f"step: next obs {tuple(td['next', 'observation'].shape)} reward_finite={finite}", flush=True)
print(f"outcome key present: {'outcome' in td['next'].keys()}", flush=True)

# throughput
N = 1024
env2 = ParkourEnvWarp(robot="spot", num_envs=N, device="cuda")
td = env2.reset()
act = torch.zeros(N, env2.robot.n_joints, device="cuda")
for _ in range(10):
    td["action"] = act
    td = env2.step(td)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    td["action"] = act
    td = env2.step(td)
torch.cuda.synchronize()
sps = 100 * N / (time.time() - t0)
print(f"throughput: {sps:,.0f} steps/sec ({N} envs)", flush=True)
print("ENV_OK", flush=True)
