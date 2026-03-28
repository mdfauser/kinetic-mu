import chex
import jax.numpy as jnp
import jax
from functools import partial

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
    game_lengths: chex.Array
    position: int
    max_games: int
    size: int

    def add_game(self, game_traj, game_length):
        # Update the nested Transition data
        new_data = jax.tree_util.tree_map(
            lambda buf_arr, traj_arr: buf_arr.at[self.position].set(traj_arr),
            self.data,
            game_traj
        )

        # set priority to maximum (1.0)
        new_priorities = self.priorities[self.position].set(1.0)
        new_lengths = self.game_lengths.at[self.position].set(game_length)
        new_pos = (self.position + 1) % self.max_games
        new_size = jnp.minimum(self.size + 1, self.max_games)

        return self.replace(data=new_data, priorities=new_priorities, game_lengths=new_lengths, position=new_pos, size=new_size)

    @partial(jax.jit, static_argnums=(2,))
    def sample_prioritized(buffer_data, key, window_size,  batch_size, alpha=0.9, beta=0.6):
        # valid priorities/occupied spots in buffer
        valid_priorities = buffer_data.priorities[:buffer_data.size]  

        # TODO scheduler for the beta 0.4 -> 1.0
        # use logits for sampling
        scaled_priorities = jnp.power(valid_priorities, alpha)
        logits = jnp.log(scaled_priorities)

        k1, k2 = jax.random.split(key)
        game_idxs = jax.random.categorical(key, k1, logits, shape=(
            batch_size, ))

        # we also use Importance Sampling to reduce the bias
        probs = valid_priorities / jnp.sum(valid_priorities)
        all_weights = jnp.power(((1.0/buffer_data.size) * (1.0/probs)), beta)
        # normalize the weights
        importance_weights = all_weights[game_idxs] / jnp.max(all_weights)

        batch_lengths = buffer_data.game_lengths[game_idxs]
        max_possible_starts = jnp.maximum(0, batch_lengths - window_size)

        k2, k_step = jax.random.split(k2)

        start_times = jax.random.randint(k_step, (batch_size,), 0, jnp.maximum(1, max_possible_starts))
        start_times = start_times.astype(jnp.int32)

        def sample_windows(start_time, game_idx):

            def get_window(array):
                return jax.lax.dynamic_slice(
                array, 
                (game_idx, start_time, 0), 
                (1, window_size, array.shape[-1])
                )

            return jax.tree_util.tree_map(get_window, buffer_data)

            # get the window for each picked sample
        batch_with_extra_dim = jax.vmap(sample_windows, in_axes=(0, 0))(
            start_times,
            game_idxs
        )
        # Clean up: Remove the "1" dimension created by the slice
        sampled_batch = jax.tree_util.tree_map(
            lambda x: x.squeeze(1), 
            batch_with_extra_dim
        )
        return sampled_batch, importance_weights


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

        return self.replace(priorities=updated_priorities_array)

    def is_ready(self, batch_size):
        """making sure to only sample from the spots which are occupied"""

        return self.size >= batch_size

    def get_status(self):
        """returns current average rewards and buffer size"""
        return jnp.average(self.data.reward), self.size


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
        game_lengths=0,
        position=0,
        max_games=1000,
        size=0
    )
