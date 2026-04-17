import jax
import jax.numpy as jnp

from search.mu_zero_mcts import MuZeroMCTS
from storage.buffer import MuZeroTransition, PrioritizedReplayBuffer, init_prioritized_buffer


class MuZero():
    def __init__(self, batch_size, num_simulations, max_seq, seq_len, obs_shape, key, unroll_steps, ):
        self.mcts = MuZeroMCTS(num_simulations=num_simulations)
        self.max_seq
        self.buffer = init_prioritized_buffer(max_seq=max_seq, seq_len=seq_len, obs_shape=obs_shape)
        self.key = key
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps

    def game_to_jax(game_list):
        stacked_traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *game_list)
        return stacked_traj

    def train(self, env, real_obs):

        # STEP 1 - Acting
        # TODO get root_value, child_visits, action_weights
        game_traj = []
        action, root_value, child_visits = self.mcts.select_action(params=None, rng_key=None,
                                          real_observation=real_obs)
        # take step in the environment
        obs, reward, terminated, info = env.step(action)
        # self.game.store(policy_output.action)
        # action_weights are the normalized visit counts of the root -> policy
        # the search_tree provides a value in the root (weighted average of all rewards and future values discovered in dream steps)
        # self.buffer.store(policy_output.action_weights, self._current_obs, self._current_action, policy_output.search_tree.summary().value)
        # transition = MuZeroTransition(
            # observation=real_obs, action=action, reward=reward, root_value=None, child_visits=None)
        transition = [real_obs, action, reward, root_value, child_visits]
        game_traj.append(transition)
        game_len = len(game_traj)
        jax_game_traj = self.game_to_jax(game_traj, self.max_seq) # TODO
        self.buffer.add_game(jax_game_traj, game_len)
        # STEP 2 - Planning
        # STEP 3 - Learning
        # TODO make unroll_steps maybe part of the obj instead of parameters
        prioritized_samples, _, importance_weights = self.buffer.prioritized_sample(self.key, self.unroll_steps, self.batch_size)
        