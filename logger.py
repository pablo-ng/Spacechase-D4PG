import tensorflow as tf
import datetime
import time

from params import Params
from utils import tf_round


class Logger(tf.Module):

    def __init__(self):
        super().__init__(name="Logger")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            self.episode_counter = tf.Variable(0, dtype=tf.int32, name="episode_counter")
            self.episode_counter_cs = tf.CriticalSection(name="episode_counter_cs")

            self.get_time = lambda: tf.reshape(tf.py_function(time.time, [], Tout=tf.float64), ()) * 1000
            self.actor_time = tf.Variable(self.get_time())
            self.learner_time = tf.Variable(self.get_time())

            self.learner_log_steps = tf.cast(Params.LEARNER_LOG_STEPS, tf.float64)
            self.actor_log_steps = tf.cast(Params.ACTOR_LOG_STEPS, tf.float64)

            ## Init Tensorboard writer
            if Params.LOG_TENSORBOARD:
                log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + \
                          f"_{Params.BUFFER_TYPE}_B{Params.MINIBATCH_SIZE}_N{Params.N_STEP_RETURNS}_{Params.NOISE_TYPE}"
                # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, write_images=False)
                self.writer = tf.summary.create_file_writer(log_dir)
            else:
                self.writer = None

    def log_ep_actor(self, n_episode, ep_steps, ep_avg_reward, noise_sigma, env_info, ep_replay_filename):
        if Params.DO_LOGGING:
            print("retracing log_ep_tensorboard")
            with tf.device(self.device):

                curr_time = self.get_time()
                avg_ep_time = (curr_time - self.actor_time.value()) / self.actor_log_steps
                self.actor_time.assign(curr_time)

                if Params.LOG_CONSOLE:
                    tf.print('E:', n_episode, '\tSteps:', ep_steps, '\t\tAvg. Reward:',
                             tf_round(ep_avg_reward, 3), '\tNoise variance:', tf_round(noise_sigma, 2), '\t', env_info)

                if Params.LOG_TENSORBOARD:
                    with self.writer.as_default():
                        step = tf.cast(n_episode, tf.int64)
                        tf.summary.scalar("Steps", ep_steps, step)
                        tf.summary.scalar("Average Reward", ep_avg_reward, step)
                        tf.cond(tf.not_equal(ep_replay_filename, ""),
                                lambda: tf.summary.text("Episode Replay", ep_replay_filename, step), lambda: False)
                        tf.summary.scalar("Avg Ep time", avg_ep_time, step)
                        self.writer.flush()

    def log_step_learner(self, n_step, td_error, priority_beta):
        if Params.DO_LOGGING:
            print("retracing log_ep_tensorboard")
            with tf.device(self.device), tf.name_scope("Logger"):

                curr_time = self.get_time()
                avg_step_time = (curr_time - self.learner_time.value()) / self.learner_log_steps
                self.learner_time.assign(curr_time)

                if Params.LOG_CONSOLE:
                    tf.print("Train step:", n_step, "\t\tTD-Error:", td_error, "\t\tPriority Beta:", priority_beta)

                if Params.LOG_TENSORBOARD:
                    with self.writer.as_default():
                        step = tf.cast(n_step, tf.int64)
                        tf.summary.scalar("TD-Error", td_error, step)
                        tf.summary.scalar("Avg Step time", avg_step_time, step)
                        self.writer.flush()

    def increment_episode(self, increment=1):
        """
        Increments counter in a thread safe manner.
        """
        with tf.device(self.device), self.name_scope:
            print("retracing Logger increment")

            def assign_add():
                return self.episode_counter.assign_add(increment).value()

            return self.episode_counter_cs.execute(assign_add)

    def n_episode(self):
        """
        Get the counter value in a thread safe manner.
        """
        with tf.device(self.device), self.name_scope:
            print("retracing Logger call")

            def get_value():
                return self.episode_counter.value()

            return self.episode_counter_cs.execute(get_value)



