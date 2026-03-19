import jax
import mctx
import jax.numpy as jnp
from inference.networks import RepresentationNet, DynamicNet, PredictionNet
from storage.game import Game
from storage.buffer import PrioritizedReplayBuffer


class MuZeroMCTS():
    def __init__(self, num_simulations, temperature=1.0):
        self.num_simulations = num_simulations
        self.temperature = temperature
        # 0 -> always picks the best move
        # 1->picks moves proportionally to how much it searched them (Training/Exploration)
        self.buffer = PrioritizedReplayBuffer()
        self.game = Game()
        self.repr_net = RepresentationNet()
        self.dyn_net = DynamicNet()
        self.pred_net = PredictionNet()

        # TODO store outside
        self._current_action = None
        self._current_obs = None
        self.evaluation_mode = False

    def root_fn(self, real_observation):
        # turns real game pixels into the first inital hidden state
        # runs once per real-world turn
        self._current_obs = real_observation  # TODO store before calling root_fn
        hidden_state = self.repr_net.forward(real_observation)

        prior_logits, value = self.pred_net.forward(hidden_state)

        return mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=hidden_state)

    def recurrent_fn(self, params, rng_key, action, embedding):
        # called on the leaf nodes and unvisited actions retrieved by the simulation step
        # returns the probability distribution of which move is actually best
        # shape of hidden_state and next_hidden_state (batch size, .. features) and normalized
        imagined_reward, imagined_next_hidden_state = self.dyn_net.forward(
            embedding, action)  # not implemented yet

        assert embedding.shape == imagined_next_hidden_state.shape

        prior_logits, value = self.pred_net.forward(imagined_next_hidden_state)

        discount = jnp.full_like(imagined_reward, fill_value=0.99)

        return mctx.RecurrentFnOutput(reward=imagined_reward, discount=discount, prior_logits=prior_logits, value=value), imagined_next_hidden_state

    def policy_out(self, params, rng_key, root):

        # TODO Before calling mctx, you set the prior_logits of illegal moves to a very large negative

        # TODO needs to play in loop against itseld
        policy_output = mctx.muzero_policy(
            params=params, rng_key=rng_key, root=root, recurrent_fn=self.recurrent_fn, num_simulations=self.num_simulations, temperature=self.temperature)

        return policy_output

    def select_action(self, params, rng_key, real_observation):
        """Pick the next action for the real world."""
        root = self.root_fn(
            real_observation=real_observation)

        prior_logits, value, embedding = root.prior_logits, root.value, root.embedding

        # noise for the prior_logits
        rng_key, rng_key = jax.random.split(rng_key)
        noise = jax.random.dirichlet(
            rng_key, jnp.ones_like(prior_logits) * 0.25)
        noisy_prior_logits = noise + prior_logits

        noisy_root = mctx.RootFnOutput(
            prior_logits=noisy_prior_logits, value=value, embedding=embedding)

        policy_out = self.policy_out(params, rng_key, noisy_root)

        visit_counts = policy_out.action_weights
        if self.evaluation_mode:
            action = jnp.argmax(visit_counts)

        else:
            # TODO Use a temperature-scaled sample
            action = jnp.random.categorical(rng_key, jnp.log(visit_counts))

        # TODO this needs to be placed somewhere outside
        # self.game.store(policy_output.action)
        # action_weights are the normalized visit counts of the root -> policy
        # the search_tree provides a value in the root (weighted average of all rewards and future values discovered in dream steps)
        # self.buffer.store(policy_output.action_weights, self._current_obs, self._current_action, policy_output.search_tree.summary().value
        #   )

        return action

    # in the training loop we are sampling a trajectory and unroll our model K steps to calculate the loss.
    # gradients need to flow from K steps all the way back to the initial representation network
