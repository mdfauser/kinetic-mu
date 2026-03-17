import jax
import mctx
import jax.numpy as jnp
from inference.networks import RepresentationNet, DynamicNet, PredictionNet


class MuZeroMCTS():
    def __init__(self, num_simulations):
        self.num_simulations = num_simulations

    def policy_out(params, rng_key, root):

        def root_fn():
            # turns real game pixels into the first inital hidden state
            pass

        def recurrent_fn(hidden_state, action):
            # called on the leaf nodes and unvisited actions retrieved by the simulation step
            # returns the probability distribution of which move is actually best
            # shape of hidden_state and next_hidden_state (batch size, .. features) and normalized

            imagined_reward, imagined_next_hidden_state = DynamicNet(
                hidden_state, action)  # not implemented yet

            policy, value = PredictionNet(imagined_next_hidden_state)

            assert hidden_state.shape == imagined_next_hidden_state.shape

            return policy

        # TODO needs to play in loop against itseld
        return mctx.muzero_policy(params=params, rng_key=rng_key, root=root, recurrent_fn=recurrent_fn(), num_simulations=self.num_simulations)

    # in the training loop we are sampling a trajectory and unroll our model K steps to calculate the loss.
    # gradients need to flow from K steps all the way back to the initial representation network
