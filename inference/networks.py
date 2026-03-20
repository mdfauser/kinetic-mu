import jax
import jax.numpy as jnp
import flax.linen as nn


class RepresentationNet(nn.Module):
    """
    Input: raw observation
    Output: hidden state
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # for images use CNN or ResNet
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        hidden_state = nn.Dense(features=self.hidden_dim)(x)

        s_min = hidden_state.min(axis=-1, keepdims=True)
        s_max = hidden_state.max(axis=-1, keepdims=True)
        return (hidden_state - s_min) / (s_max - s_min + 1e-6)


class DynamicNet(nn.Module):
    """
    Input: Hidden state and action
    Output: next hidden state and immediate reward
    """
    hidden_dim: int
    reward_dim: int
    num_actions: int

    @nn.compact
    def __call__(self, hidden_state, action):
        # for discrete
        action_one_hot = jax.nn.one_hot(action, self.num_actions)
        action_one_hot = action_one_hot.reshape((action_one_hot.shape[0], -1))
        x = jnp.concatenate([hidden_state, action_one_hot], axis=-1)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        next_hidden = nn.Dense(features=self.hidden_dim)(x)
        reward = nn.Dense(features=self.reward_dim)(x)

        s_min = next_hidden.min(axis=-1, keepdims=True)
        s_max = next_hidden.max(axis=-1, keepdims=True)
        norm_hidden = (next_hidden - s_min) / (s_max - s_min + 1e-6)

        return norm_hidden, reward


class PredictionNet(nn.Module):
    """
    Input: Hidden state
    Output: policy and value
    """
    hidden_dim: int
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        value = nn.Dense(features=1)(x)
        # for discrete actions
        policy_logits = nn.Dense(features=self.num_actions)(x)
        # softmax outside

        return policy_logits, value
