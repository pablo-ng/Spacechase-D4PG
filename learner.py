import tensorflow as tf

from params import Params
from networks import ActorNetwork, CriticNetwork, update_target


class Learner(tf.Module):

    def __init__(self, actor_event_run, actor_event_stop, replay_buffer):
        super(Learner, self).__init__(name="Learner")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            self.dtype = Params.DTYPE
            self.batch_size = Params.MINIBATCH_SIZE
            self.gamma = Params.GAMMA
            self.tau = Params.TAU
            self.replay_buffer = replay_buffer
            self.actor_event_run = actor_event_run
            self.actor_event_stop = actor_event_stop

            ## Init Networks
            self.actor = ActorNetwork(with_target_net=True)
            self.critic = CriticNetwork()

    def run(self):
        print("retracing learner run")

        with tf.device(self.device), self.name_scope:

            ## Wait for replay buffer to fill
            tf.while_loop(
                lambda _: tf.less(self.replay_buffer.size(), Params.MINIBATCH_SIZE),
                self.train_step,
                loop_vars=tf.constant(0)
            )

            ## Do training
            tf.while_loop(
                lambda *args: args[0] < Params.MAX_STEPS_TRAIN,
                self.train_step,
                loop_vars=[tf.constant(0)]
            )

            # Stop actors
            self.actor_event_stop.set()

    @tf.function()
    def train_step(self):
        print("retracing train_step")

        with tf.device(self.device), self.name_scope:

            # print("Eager Execution:  ", executing_eagerly())
            # print("Eager Keras Model:", self.actor.actor_network.run_eagerly)

            ## Get batch from replay memory as shapes (bs, space)
            (s_batch, a_batch, r_batch, t_batch, s2_batch), weights, idxes = \
                self.replay_buffer.sample_batch(self.batch_size, self.priority_beta)

            ## Predict target Q value by target critic network
            action = self.actor.target_actor_network(s2_batch, training=False)
            target_q = tf.cast(self.critic.target_critic_network([s2_batch, action], training=False), self.dtype)

            ## Compute y_i (target) for batch
            y_i = tf.add(r_batch, tf.where(t_batch, 0., self.gamma * target_q))

            ## Train the critic on given targets
            max_q, td_error = self.critic.train(x=[s_batch, a_batch], y=y_i, is_weights=weights)

            # Compute actions for state batch
            actions = self.actor.actor_network(s_batch, training=False)

            # Compute and negate action values (to enable gradient ascent)
            values = -self.critic.critic_network([s_batch, actions], training=False)

            ## Compute (dq / da * da / dtheta = dq / dtheta) grads (action values grads wrt. actor network weights)
            actor_gradients = tf.gradients(values, self.actor.tvariables)

            # Normalize grads element-wise
            actor_gradients = [tf.divide(gradient, tf.cast(self.batch_size, self.dtype)) for gradient in
                               actor_gradients]

            # Apply gradients to actor net
            self.actor.actor_network.optimizer.apply_gradients(zip(actor_gradients, self.actor.tvariables))

            # Update target networks
            update_target(self.actor.target_tvariables + self.actor.target_nvariables,
                          self.actor.tvariables + self.actor.nvariables, self.tau)
            update_target(self.critic.target_tvariables + self.critic.target_nvariables,
                          self.critic.tvariables + self.critic.nvariables, self.tau)

            # Use critic TD error to update priorities
            self.replay_buffer.update_priorities(idxes, td_error)

            # Increment beta value
            # self.priority_beta.assign_add(self.priority_beta_increment)
            # todo

            return tf.cast(max_q, self.dtype)




