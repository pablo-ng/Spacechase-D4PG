import tensorflow as tf

from params import Params
from networks import ActorNetwork, CriticNetwork, update_weights


class Learner(tf.Module):

    def __init__(self, logger, replay_buffer):
        super(Learner, self).__init__(name="Learner")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            self.dtype = Params.DTYPE
            self.logger = logger
            self.batch_size = Params.MINIBATCH_SIZE
            self.gamma = Params.GAMMA
            self.tau = Params.TAU
            self.replay_buffer = replay_buffer
            self.priority_beta = tf.Variable(Params.BUFFER_PRIORITY_BETA_START)

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
            tf.cond(
                tf.logical_not(Params.BUFFER_FROM_REVERB),
                lambda: tf.while_loop(
                    lambda: tf.less(self.replay_buffer.size(), Params.MINIBATCH_SIZE),
                    lambda: [],
                    loop_vars=[]
                ),
                lambda: [],
            )

            ## Do training
            tf.while_loop(
                lambda n_step: tf.less_equal(n_step, Params.MAX_STEPS_TRAIN),
                self.train_step,
                loop_vars=[tf.constant(1)]
            )

    def train_step(self, n_step):
        print("retracing train_step")

        with tf.device(self.device), self.name_scope:

            print("Eager Execution:  ", tf.executing_eagerly())
            print("Eager Keras Model:", self.actor.actor_network.run_eagerly)

            if Params.BUFFER_FROM_REVERB:

                ## Get batch from replay memory
                (s_batch, a_batch, _, _, s2_batch, target_z_atoms_batch), weights_batch, idxes_batch = \
                    self.replay_buffer.sample_batch(self.batch_size, self.priority_beta)

            else:

                ## Get batch from replay memory
                (s_batch, a_batch, r_batch, t_batch, s2_batch, g_batch), weights_batch, idxes_batch = \
                    self.replay_buffer.sample_batch(self.batch_size, self.priority_beta)

                ## Compute targets (bellman update)
                target_z_atoms_batch = tf.where(t_batch, 0., self.critic.target_z_atoms)
                target_z_atoms_batch = r_batch + (target_z_atoms_batch * g_batch)

            ## Predict target Q value by target critic network
            target_action_batch = self.actor.target_actor_network(s2_batch, training=False)
            target_q_probs = tf.cast(self.critic.target_critic_network([s2_batch, target_action_batch], training=False), self.dtype)

            ## Train the critic on given targets
            td_error_batch = self.critic.train(x=[s_batch, a_batch], target_z_atoms=target_z_atoms_batch, target_q_probs=target_q_probs, is_weights=weights_batch)

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
            self.replay_buffer.update_priorities(idxes_batch, td_error_batch)
            # if Params.BUFFER_TYPE in ("ReverbPrioritized",):
            #     priorities = tf.pow((tf.abs(td_error_batch) + Params.BUFFER_PRIORITY_EPSILON),
            #                         Params.BUFFER_PRIORITY_ALPHA)
            #     self.replay_client.update_priorities(tf.constant([Params.BUFFER_TYPE]), keys=idxes_batch,
            #                                          priorities=tf.cast(priorities, tf.float64))
            #     # todo can set reverb to use 32? / dtype

            # Increment beta value
            self.priority_beta.assign_add(Params.BUFFER_PRIORITY_BETA_INCREMENT)

            # Log status
            tf.cond(
                tf.equal(tf.math.mod(n_step, tf.constant(200)), tf.constant(0)),
                lambda: self.logger.log_step_learner(n_step, tf.cast(tf.reduce_mean(td_error_batch), Params.DTYPE)),
                lambda: None
            )

            return tf.add(n_step, 1)




