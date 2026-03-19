import jax
import jax.numpy as jnp
from search.mu_zero_mcts import MuZeroMCTS

if __name__ == "__main__":
    mcts = MuZeroMCTS(5)
    fake_obs = jnp.zeros((4, 8, 8, 3))

    key = jax.random.PRNGKey(42)
    out = mcts.policy_out(params=None, rng_key=key, real_observation=fake_obs)
