import tensorflow as tf
from os import environ
import threading

from params import Params
from utils import TFOrnsteinUhlenbeckActionNoise, GaussianNoise
from actor import Actor
from learner import Learner
from experience_replay_tf import PrioritizedReplayBufferProportional, PrioritizedReplayBufferRankBased, UniformReplayBuffer


def train():

    ## Set TF dtype and filter devices
    tf.keras.backend.set_floatx(Params.DTYPE)
    # environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ## Set random seed
    if Params.SEED is not None:
        environ['PYTHONHASHSEED'] = str(Params.SEED)
        tf.random.set_seed(Params.SEED)

    # ## Init Video Recorder
    # recorder = VideoRecorder()
    # record_episode = tf.Variable(Params.RECORD_VIDEO)

    ## Init Replay Buffer
    # self.priority_beta = tf.Variable(Params.BUFFER_PRIORITY_BETA_START)
    # self.priority_beta_increment = (Params.BUFFER_PRIORITY_BETA_END - Params.BUFFER_PRIORITY_BETA_START) / 130000
    buffer_data_spec = (
        tf.TensorSpec(Params.ENV_OBS_SPACE, dtype=Params.DTYPE),
        tf.TensorSpec(Params.ENV_ACT_SPACE, dtype=Params.DTYPE),
        tf.TensorSpec((1,), dtype=Params.DTYPE),
        tf.TensorSpec((1,), dtype=tf.bool),
        tf.TensorSpec(Params.ENV_OBS_SPACE, dtype=Params.DTYPE)
    )
    if Params.BUFFER_TYPE == "Uniform":
        replay_buffer = UniformReplayBuffer(buffer_data_spec)
    elif Params.BUFFER_TYPE == "Prioritized":
        replay_buffer = PrioritizedReplayBufferProportional(buffer_data_spec)
    else:
        raise Exception(f"Buffer with name {Params.BUFFER_TYPE} not found.")

    ## Init Actor-Noise
    if Params.NOISE_TYPE == "Gaussian":
        actor_noise = GaussianNoise()
    elif Params.NOISE_TYPE == "OrnsteinUhlenbeck":
        actor_noise = TFOrnsteinUhlenbeckActionNoise()
    else:
        raise Exception(f"Noise with name {Params.NOISE_TYPE} not found.")

    ## Init threads list
    threads = []

    ## Create threading events
    actor_event_run = threading.Event()
    actor_event_stop = threading.Event()

    ## Init learner
    learner = Learner(actor_event_run, actor_event_stop, replay_buffer)
    threads.append(threading.Thread(target=learner.run))

    ## Init actors
    for n_actor in range(Params.NUM_ACTORS):
        actor = Actor(n_actor, actor_event_run, actor_event_stop, replay_buffer, actor_noise)
        threads.append(threading.Thread(target=actor.run))

    ## Start all threads
    for t in threads:
        t.start()

    ## Wait for finish
    for t in threads:
        t.join()


if __name__ == '__main__':
    train()



