import tensorflow as tf
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import datetime
from os import mkdir
import imageio
import os

from params import Params


@tf.function(input_signature=[tf.TensorSpec((), tf.float32), tf.TensorSpec((), tf.int32)])
def tf_round(x, decimals=0):
    print("retracing tf_round")
    multiplier = tf.cast(10 ** decimals, dtype=x.dtype)
    return tf.cast((x * multiplier), tf.int32) / tf.cast(multiplier, tf.int32)


class TFOrnsteinUhlenbeckActionNoise(tf.Module):

    # Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

    def __init__(self):
        super(TFOrnsteinUhlenbeckActionNoise, self).__init__(name="TFOrnsteinUhlenbeckActionNoise")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            # std values: sigma=0.3, theta=.15, dt=1e-2, decay=6e-5, x0=None
            self.dtype = Params.DTYPE
            self.theta = Params.NOISE_THETA
            self.mu = tf.fill(Params.ENV_ACT_SPACE, value=Params.NOISE_MU)
            self.sigma = tf.Variable(Params.NOISE_SIGMA)
            self.sigma_min = 5e-3
            self.dt = Params.DT
            self.x0 = Params.NOISE_X0
            self.x_prev = tf.Variable(tf.zeros_like(self.mu))
            self.decay = Params.NOISE_DECAY
            self.reset()

    def __call__(self):
        print("retracing noise call")
        with tf.device(self.device), self.name_scope:
            tf.cond(tf.math.less(self.sigma, self.sigma_min), tf.no_op, lambda: self.sigma.assign(tf.math.multiply(self.sigma, self.decay), read_value=False))
            x = tf.math.add(
                tf.math.add(self.x_prev, tf.math.multiply(tf.math.multiply(self.theta, tf.math.subtract(self.mu, self.x_prev)), self.dt)),
                tf.math.multiply(tf.math.multiply(self.sigma, tf.sqrt(self.dt)), tf.random.normal(self.mu.shape))
            )
            self.x_prev.assign(x)
            return tf.cast(x, self.dtype)

    def reset(self):
        print("retracing noise reset")
        with tf.device(self.device), self.name_scope:
            self.x_prev.assign(tf.cond(tf.math.not_equal(self.x0, 0.), lambda: self.x0, lambda: tf.zeros_like(self.mu)))

    def decrease_sigma(self):
        pass


class GaussianNoise(tf.Module):

    def __init__(self):
        super(GaussianNoise, self).__init__(name="GaussianNoise")
        self.device = Params.DEVICE
        with tf.device(self.device), self.name_scope:
            self.mu = tf.fill(Params.ENV_ACT_SPACE, value=Params.NOISE_MU)
            self.shape = Params.ENV_ACT_SPACE
            self.sigma = tf.Variable(Params.NOISE_SIGMA)
            self.sigma_min = Params.NOISE_SIGMA_MIN
            self.decay = Params.NOISE_DECAY
            self.bound = Params.ENV_ACT_BOUND

    def __call__(self):
        with tf.device(self.device), self.name_scope:
            noise = tf.random.normal(self.shape) * self.sigma * self.bound + self.mu
            return noise

    def decrease_sigma(self):
        with tf.device(self.device), self.name_scope:
            tf.cond(tf.math.less(self.sigma, self.sigma_min),
                    tf.no_op, lambda: self.sigma.assign(tf.math.multiply(self.sigma, self.decay), read_value=False))


# class Logger():
#
    # def __init__(self):
    #     ## Init Tensorboard writer
    #     if Params.LOG_TENSORBOARD:
    #         # todo see write_graph and write_images
    #         log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #         tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False,
    #                                                               write_images=False)
    #         WRITER = tf.summary.create_file_writer(log_dir)
#
#     @tf.function()
#     def log_ep_tensorboard(n_episode, ep_steps, ep_avg_reward, ep_avg_q_max, ep_replay_filename):
#         print("retracing log_ep_tensorboard")
#         with WRITER.as_default():
#             step = tf.cast(n_episode, tf.int64)
#             # tf.summary.scalar("Steps", ep_steps, step)
#             tf.summary.scalar("Average Reward", ep_avg_reward, step)
#             # tf.summary.scalar("Avg. Qmax", ep_avg_q_max, step)
#             tf.cond(tf.not_equal(ep_replay_filename, ""), lambda: tf.summary.text("Episode Replay", ep_replay_filename, step), lambda: False)
#             # tf.summary.histogram("Action ditribution", ep_actions, step)
#             WRITER.flush()


class VideoRecorder:

    def __init__(self):

        if not Params.RECORD_VIDEO: return

        self.frame_size = Params.FRAME_SIZE.numpy()
        self.pad_len = 100
        self.out_type = Params.RECORD_VIDEO_TYPE

        if np.less(self.frame_size, 64).any():
            raise Exception("Frame size must be > 64px")

        self.video_writer = None
        self.writer_path = 'recorded/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        try:
            mkdir(self.writer_path)
        except OSError:
            raise "Creation of the directory %s failed" % self.writer_path

    def pad(self):
        for i in range(self.pad_len):
            if self.video_writer is not None:
                self.video_writer.write(np.zeros(self.frame_size.astype(int)))

    def save_video(self, frames, n_episode, ep_avg_reward):
        path = self.writer_path + '/episode_' + str(n_episode.numpy()) + '_avg_reward_' + str(int(ep_avg_reward))

        if self.out_type == "GIF":
            path += ".gif"
            with imageio.get_writer(path, mode='I', duration=0.04) as writer:
                for i in range(frames.shape[0]):
                    writer.append_data(frames[i].numpy())

        elif self.out_type == "MP4":
            path += ".mp4"
            self.video_writer = VideoWriter(path, VideoWriter_fourcc(*'mp4v'), 120., tuple(self.frame_size), isColor=False)  # alternative codec: MJPG

            for i in range(frames.shape[0]):
                self.video_writer.write(frames[i].numpy())

            self.pad()
            self.video_writer.release()

        self.video_writer = None

        print("saved video ...")

        return os.path.abspath(path)







