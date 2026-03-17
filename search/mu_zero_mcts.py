import jax
import mctx
import jax.numpy as jnp
from inference.networks import RepresentationNet, DynamicNet, PredictionNet


class MuZeroMCTS():
    def __init__(self, num_simulations):
        self.num_simulations = num_simulations

    def root_fn(self, real_observation):
        # turns real game pixels into the first inital hidden state
        # runs once per real-world turn
        hidden_state = RepresentationNet(real_observation)

        prior_logits, value = PredictionNet(hidden_state)

        return prior_logits, value, hidden_state  # RootFnOutput

    def recurrent_fn(self, hidden_state, action):
        # called on the leaf nodes and unvisited actions retrieved by the simulation step
        # returns the probability distribution of which move is actually best
        # shape of hidden_state and next_hidden_state (batch size, .. features) and normalized

        imagined_reward, imagined_next_hidden_state = DynamicNet(
            hidden_state, action)  # not implemented yet

        assert hidden_state.shape == imagined_next_hidden_state.shape

        prior_logits, value = PredictionNet(imagined_next_hidden_state)

        discount = jnp.full_like(imagined_reward, fill_value=0.99)

        return imagined_reward, discount, prior_logits, value,  # RecurrentFnOutput

    def policy_out(self, params, rng_key):
        # TODO needs to play in loop against itseld
        return mctx.muzero_policy(params=params, rng_key=rng_key, root=self.root_fn, recurrent_fn=self.recurrent_fn, num_simulations=self.num_simulations)

    # in the training loop we are sampling a trajectory and unroll our model K steps to calculate the loss.
    # gradients need to flow from K steps all the way back to the initial representation network
