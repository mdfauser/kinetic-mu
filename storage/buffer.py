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
    max_size: int
    size: int

    def add_trajectory(self, new_traj, slot):
        # Update the nested Transition data
        new_data = jax.tree_util.tree_map(
            lambda buf_arr, traj_arr: buf_arr.at[slot].set(traj_arr),
            self.data,
            new_traj
        )

        # set priority to maximum (1.0)
        new_priorities = self.prioritiesat[slot].set(1.0)

        new_pos = (self.position + 1) % self.max_size
        new_size = jnp.minimum(self.size + 1, self.max_size)

        return self.replace(data=new_data, priorities=new_priorities, position=new_pos, size=new_size)

    def prioritized_sample(self, key, batch_size, alpha=0.9, beta=0.6):
        # TODO scheduler for the beta 0.4 -> 1.0
        # use logits for sampling
        scaled_priorities = jnp.power(self.priorities, alpha)
        logits = jnp.log(scaled_priorities)
        # we also use Importance Sampling to reduce the bias
        probs = self.priorities / jnp.sum(self.priorities)
        importance_weights = jnp.power(((1.0/self.size) * (1.0/probs)), beta)
        # normalize the weights
        importance_weights = importance_weights / jnp.max(importance_weights)

        idx = jax.random.categorical(key, self.size, logits, shape=(
            batch_size, ))

        return jax.tree_util.tree_map(lambda x: x[idx], self.data), idx, importance_weights

    def compute_n_step_targets(self, rewards, values, n_steps, gamma):
        """Calculate the N-step returns."""
        # TODO add mask to cut off steps to ensure rewards and values are 0 after the game ended
        bootstrap_values = jnp.roll(values, -n_steps)

        discounted_bootstrap = (gamma ** n_steps) * bootstrap_values

        kernel = gamma ** jnp.arange(n_steps)

        padded_rewards = jnp.concatenate([rewards, jnp.zeros(n_steps-1)])

        reward_sums = jax.scipy.signal.convolve(
            padded_rewards,
            kernel[::-1],
            mode='valid'
        )

        return reward_sums[:len(rewards)] + discounted_bootstrap

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        new_priorities = jnp.abs(td_errors) + epsilon
        updated_priorities_array = self.priorities.at[indices].set[new_priorities]

        return self.replace(selfpriorities=updated_priorities_array)

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
        position=0,
        max_size=1000,
        size=0
    )
