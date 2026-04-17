from collections import deque
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from inference.mu_zero import MuZero

env = gym.make("CartPole-v1", render_mode=None)

obs_shape = env.observation_space.shape
key, subkey = jax.random.PRNGKey(42)

# model = MuZero(batch_size=16, num_simulations=10, max_seq=100, seq_len=20, obs_shape=obs_shape, key=key, unroll_steps=key)

train_ppo = False
if train_ppo:
    print("start trainign PPO ...")
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
    model.learn(total_timesteps=500000)

obs = env.reset()

successes = deque([], maxlen=100)
done = False
list_rewards = []

print("start training")
for e in range(1000):
    rewards = 0
    while not(done):
        action = np.random.randint(1, size=1)
        # model.train(env) #TODO implement the training loop
        obs, reward, done, info, _ = env.step(action[0])
        rewards += reward

    obs = env.reset()
    list_rewards.append(rewards)

print(f"total episodes: {e} | avg return:{sum(list_rewards)/100} ")

env.close()

# if __name__ == "__main__":
#     train()
