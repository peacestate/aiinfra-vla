import time, numpy as np, gymnasium as gym, gym_aloha  # noqa: F401

env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
obs, info = env.reset(seed=0)
print("obs keys:", list(obs.keys()))
print("agent_pos:", np.asarray(obs["agent_pos"]).shape, "(14 = 2 arms x 7 joints)")
for k, v in obs["pixels"].items():
    print(f"  cam {k}: {np.asarray(v).shape}")
print("action space:", env.action_space.shape)

t0 = time.perf_counter()
n = 20
for _ in range(n):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
dt = (time.perf_counter() - t0) / n
print(f"sim+render: {dt*1000:.1f} ms/step ({1/dt:.1f} Hz)")
env.close()
print("SMOKE_OK")
