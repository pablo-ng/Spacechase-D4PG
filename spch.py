import tensorflow as tf
from os import environ
import threading
import datetime

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
    priority_beta = tf.Variable(Params.BUFFER_PRIORITY_BETA_START)
    buffer_data_spec = (
        tf.TensorSpec(Params.ENV_OBS_SPACE, dtype=Params.DTYPE),    # state
        tf.TensorSpec(Params.ENV_ACT_SPACE, dtype=Params.DTYPE),    # action
        tf.TensorSpec((1,), dtype=Params.DTYPE),                    # reard
        tf.TensorSpec((1,), dtype=tf.bool),                         # terminal
        tf.TensorSpec(Params.ENV_OBS_SPACE, dtype=Params.DTYPE),    # state2
        tf.TensorSpec((1,), dtype=Params.DTYPE),                    # gamma**N
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

    ## Create threading event
    actor_event_stop = threading.Event()

    ## Init learner
    learner = Learner(actor_event_stop, replay_buffer, priority_beta)
    threads.append(threading.Thread(target=learner.run))
    learner_policy_variables = learner.actor.tvariables + learner.actor.nvariables

    ## Init actors
    # for n_actor in range(Params.NUM_ACTORS):
    actor1 = Actor(1, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise)
    threads.append(threading.Thread(target=actor1.run))

    # actor2 = Actor(2, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise)
    # threads.append(threading.Thread(target=actor2.run))
    #
    # actor3 = Actor(3, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise)
    # threads.append(threading.Thread(target=actor3.run))
    #
    # actor4 = Actor(4, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise)
    # threads.append(threading.Thread(target=actor4.run))

    # actor5 = Actor(5, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise, writer)
    # threads.append(threading.Thread(target=actor5.run))
    #
    # actor6 = Actor(6, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise, writer)
    # threads.append(threading.Thread(target=actor6.run))
    #
    # actor7 = Actor(7, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise, writer)
    # threads.append(threading.Thread(target=actor7.run))
    #
    # actor8 = Actor(8, actor_event_stop, learner_policy_variables, replay_buffer, actor_noise, writer)
    # threads.append(threading.Thread(target=actor8.run))

    ## Start all threads
    for t in threads:
        t.start()

    ## Wait for finish
    for t in threads:
        t.join()


if __name__ == '__main__':
    train()



