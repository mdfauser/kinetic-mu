import jax
import mctx
import jax.numpy as jnp
from inference.networks import RepresentationNet, DynamicNet, PredictionNet
from storage.game import Game
from storage.buffer import PrioritizedReplayBuffer


class MuZeroMCTS():
    def __init__(self, num_simulations, temperature):
        self.num_simulations = num_simulations
        self.temperature = self.temperature
        # 0 -> always picks the best move
        # 1->picks moves proportionally to how much it searched them (Training/Exploration)
        self.buffer = PrioritizedReplayBuffer()
        self.game = Game()

    def root_fn(self, real_observation):
        # turns real game pixels into the first inital hidden state
        # runs once per real-world turn
        hidden_state = RepresentationNet(real_observation)

        prior_logits, value = PredictionNet(hidden_state)

        return mctx.RootFnOutput(prior_logits, value, hidden_state)

    def recurrent_fn(self, hidden_state, action):
        # called on the leaf nodes and unvisited actions retrieved by the simulation step
        # returns the probability distribution of which move is actually best
        # shape of hidden_state and next_hidden_state (batch size, .. features) and normalized

        imagined_reward, imagined_next_hidden_state = DynamicNet(
            hidden_state, action)  # not implemented yet

        assert hidden_state.shape == imagined_next_hidden_state.shape

        prior_logits, value = PredictionNet(imagined_next_hidden_state)

        discount = jnp.full_like(imagined_reward, fill_value=0.99)

        return mctx.RecurrentFnOutput(imagined_reward, discount, prior_logits, value)

    def policy_out(self, params, rng_key, real_observation):

        root = self.root_fn(
            real_observation=real_observation)

        prior_logits, value, embedding = root.prior_logits, root.value, root.embedding

        # noise for the prior_logits
        rng_key, rng_key = jax.random.split(rng_key)
        noise = jax.random.dirichlet(
            rng_key, jnp.ones_like(prior_logits) * 0.25)
        noisy_prior_logits += noise

        noisy_root = mctx.RootFnOutput(noisy_prior_logits, value, embedding)

        # TODO needs to play in loop against itseld
        policy_output = mctx.muzero_policy(
            params=params, rng_key=rng_key, root=noisy_root, recurrent_fn=self.recurrent_fn, num_simulations=self.num_simulations, temperature=self.temperature)

        self.game.store(policy_output.action)
        self.buffer.store(policy_output.action_weights,
                          policy_output.search_tree)

        return policy_output

    # in the training loop we are sampling a trajectory and unroll our model K steps to calculate the loss.
    # gradients need to flow from K steps all the way back to the initial representation network
