import jax
import haiku as hk
import jax.numpy as jnp


class RepresentationNet():
    def __init__(self):
        pass

    def forward(self, obs):
        # return zero-tensor for testing
        batch_size = obs.shape[0]
        return jnp.zeros((batch_size, 16))


class DynamicNet():

    def __init__(self):
        pass

    def forward(self, embedding, action):
        batch_size = embedding.shape[0]
        # return zero-tensor
        return jnp.zeros((batch_size,)), jnp.zeros((batch_size, 16))


class PredictionNet():

    def __init__(sefl):
        pass

    def forward(self, embedding):
        batch_size = embedding.shape[0]
        num_actions = 4  # Let's assume 4 possible moves

        # Returns [Batch, Logits] and [Batch, Value]
        logits = jnp.zeros((batch_size, num_actions))
        value = jnp.zeros((batch_size,))
        return logits, value
