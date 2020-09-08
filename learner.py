import tensorflow as tf

from params import Params
from utils import Logger
from networks import ActorNetwork, CriticNetwork, update_weights


class Learner(tf.Module):

    def __init__(self, actor_event_stop, replay_buffer, priority_beta):
        super(Learner, self).__init__(name="Learner")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            self.dtype = Params.DTYPE
            self.batch_size = Params.MINIBATCH_SIZE
            self.gamma = Params.GAMMA
            self.tau = Params.TAU
            self.replay_buffer = replay_buffer
            self.priority_beta = priority_beta
            self.actor_event_stop = actor_event_stop

            ## Init Networks
            self.actor = ActorNetwork(with_target_net=True)
            self.critic = CriticNetwork()

            # Save shared variables
            self.policy_variables = self.actor.tvariables + self.actor.nvariables

    @tf.function()
    def run(self):
        print("retracing learner run")

        with tf.device(self.device), self.name_scope:

            ## Wait for replay buffer to fill
            tf.while_loop(
                lambda: tf.less(self.replay_buffer.size(), Params.MINIBATCH_SIZE),
                lambda: [],
                loop_vars=[]
            )

            ## Do training
            n_steps = tf.while_loop(
                lambda n_step: tf.less_equal(n_step, Params.MAX_STEPS_TRAIN),
                self.train_step,
                loop_vars=[tf.constant(1)]
            )

            ## Stop actors
            tf.cond(tf.greater_equal(n_steps, Params.MAX_STEPS_TRAIN), lambda: tf.py_function(self.actor_event_stop.set, inp=[], Tout=[]), tf.no_op)

    def train_step(self, n_step):
        print("retracing train_step")

        with tf.device(self.device), self.name_scope:

            print("Eager Execution:  ", tf.executing_eagerly())
            print("Eager Keras Model:", self.actor.actor_network.run_eagerly)

            ## Get batch from replay memory as shapes (bs, space)
            (s_batch, a_batch, r_batch, t_batch, s2_batch, g_batch), weights_batch, idxes_batch = \
                self.replay_buffer.sample_batch(self.batch_size, self.priority_beta)

            ## Predict target Q value by target critic network
            target_action_batch = self.actor.target_actor_network(s2_batch, training=False)
            target_q_probs = tf.cast(self.critic.target_critic_network([s2_batch, target_action_batch], training=False), self.dtype)

            ## Compute targets (bellman update)
            target_z_atoms_batch = tf.where(t_batch, 0., self.critic.target_z_atoms)
            target_z_atoms_batch = r_batch + (target_z_atoms_batch * g_batch)

            ## Train the critic on given targets
            td_error = self.critic.train(x=[s_batch, a_batch], target_z_atoms=target_z_atoms_batch, target_q_probs=target_q_probs, is_weights=weights_batch)

            # Compute actions for state batch
            actions = self.actor.actor_network(s_batch, training=False)

            # Compute and negate action values (to enable gradient ascent)
            values = self.critic.critic_network([s_batch, actions], training=False)

            ## Compute (dq / da * da / dtheta = dq / dtheta) grads (action values grads wrt. actor network weights)
            action_grads = tf.gradients(values, actions, self.critic.z_atoms)[0]
            actor_gradients = tf.gradients(actions, self.actor.tvariables, -action_grads)

            # Normalize grads element-wise
            actor_gradients = [tf.divide(gradient, tf.cast(self.batch_size, self.dtype)) for gradient in actor_gradients]

            # Apply gradients to actor net
            self.actor.actor_network.optimizer.apply_gradients(zip(actor_gradients, self.actor.tvariables))

            # Update target networks
            update_weights(self.actor.target_tvariables + self.actor.target_nvariables,
                           self.actor.tvariables + self.actor.nvariables, self.tau)
            update_weights(self.critic.target_tvariables + self.critic.target_nvariables,
                           self.critic.tvariables + self.critic.nvariables, self.tau)

            # Use critic TD error to update priorities
            self.replay_buffer.update_priorities(idxes_batch, td_error)

            # Increment beta value
            self.priority_beta.assign_add(Params.BUFFER_PRIORITY_BETA_INCREMENT)

            # Log status
            tf.cond(
                tf.equal(tf.math.mod(n_step, tf.constant(200)), tf.constant(0)),
                lambda: Logger.log_step_learner(n_step, tf.cast(tf.reduce_mean(td_error), Params.DTYPE)),
                lambda: None
            )

            return tf.add(n_step, 1)




