import tensorflow as tf
import datetime

from params import Params
from utils import tf_round


class Logger(tf.Module):

    def __init__(self):
        super().__init__(name="Logger")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            self.episode_counter = tf.Variable(0, dtype=tf.int32, name="episode_counter")
            self.episode_counter_cs = tf.CriticalSection(name="episode_counter_cs")

            ## Init Tensorboard writer
            if Params.LOG_TENSORBOARD:
                # todo see write_graph and write_images
                log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, write_images=False)
                self.writer = tf.summary.create_file_writer(log_dir)
            else:
                self.writer = None

    def log_ep_actor(self, ep_steps, ep_avg_reward, noise_sigma, env_info, ep_replay_filename):
        print("retracing log_ep_tensorboard")
        with tf.device(self.device), tf.name_scope("Logger"):

            if Params.LOG_CONSOLE:
                tf.print('E:', self.n_episode(), '\tSteps:', ep_steps, '\t\tAvg. Reward:', tf_round(ep_avg_reward, 3),
                         '\tNoise variance:', tf_round(noise_sigma, 2), '\t', env_info)

            if Params.LOG_TENSORBOARD:
                with self.writer.as_default():
                    step = tf.cast(self.n_episode(), tf.int64)
                    tf.summary.scalar("Steps", ep_steps, step)
                    tf.summary.scalar("Average Reward", ep_avg_reward, step)
                    tf.cond(tf.not_equal(ep_replay_filename, ""),
                            lambda: tf.summary.text("Episode Replay", ep_replay_filename, step), lambda: False)
                    self.writer.flush()
            else:
                return None

    def log_step_learner(self, n_step, td_error):
        print("retracing log_ep_tensorboard")
        with tf.device(self.device), tf.name_scope("Logger"):

            if Params.LOG_CONSOLE:
                tf.print("Train step:", n_step, "\t\tTD-Error:", td_error)

            if Params.LOG_TENSORBOARD:
                with self.writer.as_default():
                    step = tf.cast(n_step, tf.int64)
                    tf.summary.scalar("TD-Error", td_error, step)
                    self.writer.flush()
            else:
                return None

    def increment_episode(self, increment=1):
        with tf.device(self.device), self.name_scope:
            print("retracing Logger increment")

            # Increments counter in a thread safe manner.

            def assign_add():
                return self.episode_counter.assign_add(increment).value()

            return self.episode_counter_cs.execute(assign_add)

    def n_episode(self):
        with tf.device(self.device), self.name_scope:
            print("retracing Logger call")

            # Get the counter value in a thread safe manner.

            def get_value():
                return self.episode_counter.value()

            return self.episode_counter_cs.execute(get_value)



