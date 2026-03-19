import chex
import jax.numpy as jnp
import jax


@chex.dataclass
class MuZeroTransition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    root_value: chex.Array
    child_visits: chex.Array

# needs to store (Observation, Action, Reward, Search Policy, Search Value)


@chex.dataclass
class PrioritizedReplayBuffer:
    data: MuZeroTransition
    priorities: chex.Array
    position: int

    def add_trajectory(self, new_traj, slot):
        # Update the nested Transition data
        new_data = jax.tree_util.tree_map(
            lambda buf_arr, traj_arr: buf_arr.at[slot].set(traj_arr),
            self.data,
            new_traj
        )
        # Return a new buffer object with the updated data
        return self.replace(data=new_data)

    def sample(self, key, batch_size):
        probs = self.priorities / jnp.sum(self.priorities)
        idx = jax.random.choice(key, self.size, shape=(
            batch_size, ), p=probs[:self.size])
        return jax.tree_util_map(lambda x: x[idx], self.data)

    def compute_n_step_targets(self, n_steps, disocunt_factor):
        """Calculate the N-step returns."""
        pass
        # n_step_return = jax.tree_util.tree_map(
        #     lambda reward, value:
        # )

    def update_priorities(self, indices, new_td_errors):
        pass

    def smaple_windows(self, key, batch_size, window_size):
        """picking random row and start time within that row"""

        pass

    def is_ready(self, min_size):
        """making sure to only sample from the spots which are occupied"""
        pass


def init_prioritized_buffer(max_seq, seq_len, obs_shape):
    buffer_data = MuZeroTransition(
        observation=jnp.zeros((max_seq, seq_len, *obs_shape)),
        action=jnp.zeros(
            (max_seq, seq_len), dtype=jnp.int32),
        reward=jnp.zeros((max_seq, seq_len)),
        value=jnp.zeros((max_seq, seq_len))
    )
    return PrioritizedReplayBuffer(
        data=buffer_data,
        priorities=jnp.zeros((max_seq,)),
        position=0
    )
