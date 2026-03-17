from collections import deque

import gymnasium as gym
import torch
import numpy
from stable_baselines3 import PPO
import supersuit as ss

from pettingzoo.butterfly import pistonball_v6

# env = pistonball_v6.parallel_env(
#     n_pistons=3, render_mode=None, continuous=False)
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v1(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(
#     env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

# env = ss.normalize_obs_v0(env)
# env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = gym.make("CartPole-v1", render_mode="rgb_array")


model = PPO(
    "CnnPolicy",
    env,
    learning_rate=1e-4,       # Much lower than default
    n_steps=1024,             # Larger buffer to average out noise
    batch_size=64,            # Smaller batch for 3-agent focus
    gae_lambda=0.95,          # Standard for stability
    target_kl=0.01,           # Forces the update to stay small
    verbose=1
)

print("start learning ...")
model.learn(total_timesteps=500000)
print("end learning")

obs = env.reset()

successes = deque([], maxlen=100)
print("start training")
for e in range(1000):
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, info = env.step(action)
    if terminated.any():
        obs = env.reset()

    if "terminal_observation" in info[0]:
        final_reward = info[0].get("episode", {}).get("r", 0)
        is_success = 1 if final_reward > 90 else 0
        successes.append(is_success)
        print(f"Episode Ended. Success: {is_success}")

    if e % 100 == 0:
        print(f"total episodes: {e} | success rate:{sum(successes)} ")

env.close()

# if __name__ == "__main__":
#     train()
