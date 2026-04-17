import numpy as np
import jax
import jax.numpy as jnp
from search.mu_zero_mcts import MuZeroMCTS
from storage.buffer import init_prioritized_buffer, MuZeroTransition


def game_to_jax(game_list):
    stacked_traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *game_list)
    return stacked_traj

if __name__ == "__main__":
    # mcts = MuZeroMCTS(5)
    # fake_obs = jnp.zeros((4, 8, 8, 3))

    # key = jax.random.PRNGKey(42)
    # out = mcts.policy_out(params=None, rng_key=key, real_observation=fake_obs)


    # test buffer
    max_seq, seq_len, A = 100, 10, 4
    obs_shape = (2, 2)
    buffer = init_prioritized_buffer(max_seq, seq_len, obs_shape, A)

    real_obs = np.zeros(obs_shape)
    action = np.ones(A)
    reward = np.array(0.0)
    root_value = np.array(0.0)
    child_visits = np.zeros(A)

    game_list = []
    for _ in range(seq_len):
        t = MuZeroTransition(
            observation=real_obs, 
            action=action, 
            reward=reward, 
            root_value=root_value, 
            child_visits=child_visits
        )
        game_list.append(t)

    jax_game = game_to_jax(game_list)

    new_buffer = buffer.add_game(jax_game, seq_len)

    # test update priorities
    indicies = np.zeros(2, dtype=int)
    indicies[0] = 5
    indicies[1] = 1
    later_buffer = new_buffer.update_priorities(indicies, indicies)
    print(later_buffer.priorities)