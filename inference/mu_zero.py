import jax
import jax.numpy as jnp

from search.mu_zero_mcts import MuZeroMCTS
from storage.buffer import MuZeroTransition, PrioritizedReplayBuffer


class MuZero():
    def __init__(self, num_simulations):
        self.mcts = MuZeroMCTS(num_simulations=num_simulations)

    def train(self, env, real_obs):

        # STEP 1 - Acting
        # TODO get root_value, child_visits, action_weights
        action, = self.mcts.select_action(params=None, rng_key=None,
                                          real_observation=real_obs)
        # take step in the environment
        obs, reward, terminated, info = env.step(action)
        # self.game.store(policy_output.action)
        # action_weights are the normalized visit counts of the root -> policy
        # the search_tree provides a value in the root (weighted average of all rewards and future values discovered in dream steps)
        # self.buffer.store(policy_output.action_weights, self._current_obs, self._current_action, policy_output.search_tree.summary().value)
        transition = MuZeroTransition(
            observation=real_obs, action=action, reward=reward, root_value=None, child_visits=None)
        # TODO add transition into game trajectory, implement for buffer add single steps of the game at a time
        # STEP 2 - Planning

        # STEP 3 - Learning
