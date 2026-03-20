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

    def prioritized_sample(self, key, unroll_steps,  batch_size, alpha=0.9, beta=0.6):
        # valid priorities/occupied spots in buffer
        valid_priorities = self.priorities[:self.size]

        # TODO scheduler for the beta 0.4 -> 1.0
        # use logits for sampling
        scaled_priorities = jnp.power(valid_priorities, alpha)
        logits = jnp.log(scaled_priorities)

        k1, k2 = jax.random.split(key)
        game_idx = jax.random.categorical(key, k1, logits, shape=(
            batch_size, ))

        # we also use Importance Sampling to reduce the bias
        probs = valid_priorities / jnp.sum(valid_priorities)
        all_weights = jnp.power(((1.0/self.size) * (1.0/probs)), beta)
        # normalize the weights
        importance_weights = all_weights[game_idx] / jnp.max(all_weights)

        batch_lengths = self.game_lengths[game_idx]
        max_possible_starts = jnp.maximum(0, batch_lengths - unroll_steps)

        start_times = jax.random.uniform(k2, shape=(
            batch_size, )) * max_possible_starts
        start_times = start_times.astype(jnp.int32)

        return jax.tree_util.tree_map(lambda x: x[start_times], self.data), start_times, importance_weights

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

    def get_window(self, start_idx, window_size):
        """picking random row and start time within that row"""

        return jax.lax.dynamic_slice_in_dim(self.data, start_idx, window_size, axis=0)

    def sample_windows(self, start_idx, window_size, key, batch_size, alpha=0.9, beta=0.6):
        prioritized_samples, start_idx, importance_weights = self.prioritized_sample(
            key, window_size, batch_size, alpha, beta)

        # get the window for each picked sample
        return jax.vmap(self.get_window, in_axes=(0, 0, None))(
            self.data,
            start_idx,
            window_size=window_size
        )

    def is_ready(self, batch_size):
        """making sure to only sample from the spots which are occupied"""

        return self.size >= batch_size


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
