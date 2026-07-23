"""Validate that MuJoCo-Warp loads and steps the parkour Spot model on this GPU.

Isolates the model-vs-Warp question: imports only warp + mujoco_warp + the scene
builder (no torch/torchrl). First run JIT-compiles kernels (slow, then cached).
"""

import time

import warp as wp

wp.init()
print("warp cuda devices:", wp.get_cuda_devices(), flush=True)

import mujoco_warp as mjw  # noqa: E402

from automataleague.envs.parkour.scene import build_parkour_model  # noqa: E402

t = time.time()
model, info = build_parkour_model("spot")
print(f"model built: nq={model.nq} nu={model.nu} ngeom={model.ngeom} "
      f"({time.time() - t:.1f}s)", flush=True)

t = time.time()
mjw_model = mjw.put_model(model)
print(f"put_model: {time.time() - t:.1f}s", flush=True)

t = time.time()
data = mjw.make_data(model, nworld=64)
print(f"make_data(nworld=64): {time.time() - t:.1f}s", flush=True)

print("stepping (first step JIT-compiles kernels)...", flush=True)
t = time.time()
for i in range(5):
    mjw.step(mjw_model, data)
    wp.synchronize()
    print(f"  step {i} done, cumulative {time.time() - t:.1f}s", flush=True)

q = data.qpos.numpy()
print(f"STEPPED OK — qpos {q.shape}, base_z[0]={float(q[0, 2]):.3f}", flush=True)
print("DONE", flush=True)
