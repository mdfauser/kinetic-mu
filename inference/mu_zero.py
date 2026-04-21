import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from search.mu_zero_mcts import MuZeroMCTS
from storage.buffer import MuZeroTransition, PrioritizedReplayBuffer, init_prioritized_buffer
from inference.networks import RepresentationNet, DynamicNet, PredictionNet


class MuZero():
    def __init__(self, batch_size, num_simulations, max_seq, seq_len, obs_shape, key, unroll_steps, ):
        self.mcts = MuZeroMCTS(num_simulations=num_simulations)
        self.max_seq
        self.buffer = init_prioritized_buffer(max_seq=max_seq, seq_len=seq_len, obs_shape=obs_shape)
        self.key = key
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        # nets
        self.h = RepresentationNet()
        self.f = PredictionNet()
        self.g = DynamicNet()
        
        optimizer = optax.adam(learning_rate=1e-3)


    def game_to_jax(game_list):
        stacked_traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *game_list)
        return stacked_traj

    def init_train_state(self, dummy_obs, dummy_action, dummy_state, optimizer):

        key = jax.random.PRNGKey(0)
        key, k_rep, k_dyn, k_pred = jax.random.split(key, 4)

        rep_params = RepresentationNet().init(k_rep, dummy_obs)
        dyn_params = DynamicNet().inint(k_dyn, dummy_state, dummy_action)
        pred_params = PredictionNet().init(k_pred, dummy_state)

        params = {
            'h': rep_params,
            'g': dyn_params,
            'f': pred_params,
        }
        return MuZeroTrainState.create(
            apply_fn=None, # We have multiple apply_fns, so we handle them manually
            params=params,
            target_params=params, # For Target Networks if you use them
            tx=optimizer,
            steps=0,
        )
    
    def self_play(self, key, params, env, real_obs):

        game_traj = []
        done = False
        while not(done):
            action, root_value, child_visits = self.mcts.select_action(params=params, rng_key=key,
                                            real_observation=real_obs)
            # take step in the environment
            real_obs, reward, done, info = env.step(action)
            # self.game.store(policy_output.action)
            # action_weights are the normalized visit counts of the root -> policy
            # the search_tree provides a value in the root (weighted average of all rewards and future values discovered in dream steps)
            transition = MuZeroTransition(
                observation=real_obs,
                action=action,
                reward=reward,
                root_value=root_value,
                child_visits=child_visits,
            )
        # outside loop
        game_traj.append(transition)
        jax_game_traj = self.game_to_jax(game_traj)

        return jax_game_traj, len(game_traj)

    # @jit
    def train_step(self, params, opt_state, samples, key):
        pass

    def rewrad_loss(r_pred, r_target):
        return jnp.mean(jnp.square(r_pred - r_target))

    def value_loss(v_pred, v_target):
        return jnp.mean(jnp.square(v_pred - v_target))

    def policy_loss(logits, pi_target):
        return jnp.mean(-jnp.sum(pi_target, jnp.softmax(logits), axis=1))

    def total_loss(self, params, samples):

        # loss at the root (no reward at root)
        s0 = self.h(params['h'], samples.observation)
        v0, pi0 = self.f(params['f'], s0)
        loss = self.value_loss(v0, samples.root_value[:,0])
        loss += self.policy_loss(pi0, samples.child_visits)

        # using lax scan
        def get_step_loss(s, xs):
            action, reward, root_value, child_visits = xs

            ns = self.h(params['h'], s)
            nv, npi = self.f(params['f'], ns)
            ns, r = self.g(params['g'], ns) 

            step_loss = self.value_loss(nv, root_value) + self.policy_loss(npi, child_visits) + self.rewrad_loss(r, reward)

            return step_loss, ns

        jax.lax.scan(get_step_loss, s0, xs=
        (samples.action, samples.reward, samples.root_value, samples.child_visits))

    def train(self, env, real_obs, key,params, opt_state):

        # STEP 1 - Acting
        jax_game_traj, game_len = self.self_play(key, params, env, real_obs)
        self.buffer = self.buffer.add_game(jax_game_traj, game_len)

        # sample from buffer
        prioritized_samples, _, importance_weights = self.buffer.prioritized_sample(self.key, self.unroll_steps, self.batch_size)

        # Training Step
        (params, opt_state), loss = self.train_step(params, opt_state, prioritized_samples, key)
        key = jax.random.split(key)

class MuZeroTrainState(train_state.TrainState):
    # You can add extra stuff here if needed, like target_params
    target_params: jax.Array
    step: int