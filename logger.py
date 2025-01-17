import tensorflow as tf
import time

from params import Params
from utils import tf_round, Counter


class Logger(tf.Module):

    def __init__(self, log_dir):
        super().__init__(name="Logger")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            self.episode_counter = Counter("episode_counter", start=-1, dtype=tf.int32)
            self.actor_steps_counter = Counter("actor_steps_counter", start=0, dtype=tf.int32)

            self.get_time = lambda: tf.reshape(tf.py_function(time.time, [], Tout=tf.float64), ()) * 1000
            self.actor_time = tf.Variable(self.get_time())
            self.learner_time = tf.Variable(self.get_time())

            self.learner_log_steps = tf.cast(Params.LEARNER_LOG_STEPS, tf.float64)

            # Init Tensorboard writer
            if Params.LOG_TENSORBOARD:
                self.log_dir = log_dir
                self.writer = tf.summary.create_file_writer(self.log_dir)
            else:
                self.writer = None

            # Log parameters
            self.log_params()

    def log_params(self):
        if Params.DO_LOGGING:
            print("retracing log_params")
            with tf.device(self.device):

                params = [f"{attr}: {getattr(Params, attr)}" for attr in dir(Params) if not attr.startswith("__")]
                params_string = "\n".join(params)

                if Params.LOG_CONSOLE:
                    tf.print("\n\n"+params_string+"\n\n")

                if Params.LOG_TENSORBOARD:
                    with self.writer.as_default():
                        tf.summary.text("Params", params_string, step=0)

    def log_ep_actor(self, n_episode, ep_steps, ep_avg_reward, ep_reward_sum_discounted,
                     noise_sigma, env_info, ep_replay_filename):
        if Params.DO_LOGGING:
            print("retracing log_ep_tensorboard")
            with tf.device(self.device):

                curr_time = self.get_time()
                actor_steps = tf.cast(self.actor_steps_counter.reset(), tf.float64)
                avg_step_time = (curr_time - self.actor_time.value()) / actor_steps
                self.actor_time.assign(curr_time)

                if Params.LOG_CONSOLE:
                    tf.print('E:', n_episode, '\tSteps:', ep_steps, '\t\tAvg. Reward:',
                             tf_round(ep_avg_reward, 3), '\tNoise variance:', tf_round(noise_sigma, 2), '\t', env_info)

                if Params.LOG_TENSORBOARD:
                    with self.writer.as_default():
                        step = tf.cast(n_episode, tf.int64)
                        tf.summary.scalar("Steps", ep_steps, step)
                        tf.summary.scalar("Average Reward", ep_avg_reward, step)
                        tf.summary.scalar("Reward Sum Discounted", ep_reward_sum_discounted, step)
                        # tf.cond(tf.not_equal(ep_replay_filename, ""),
                        #         lambda: tf.summary.text("Episode Replay", ep_replay_filename, step), lambda: False)
                        tf.summary.scalar("Avg step time", avg_step_time, step)
                        tf.summary.scalar("Noise variance", noise_sigma, step)
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





