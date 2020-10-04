import tensorflow as tf
from os import environ
import threading

from params import Params
from learner import Learner
from actor import Actor

from experience_replay import UniformReplayBuffer
from experience_replay_reverb import ReverbUniformReplayBuffer, ReverbPrioritizedReplayBuffer
from noise import TFOrnsteinUhlenbeckActionNoise, GaussianNoise
from video_recorder import VideoRecorder
from logger import Logger

"""
TODO
- logger: see write_graph and write_images
- analytical solution, train agent only for control / for approaching x/y coords
- name all operations, then can use profiler: 
    https://www.tensorflow.org/guide/profiler#profiling_apis
    https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
- Learning from demonstrations
    - Deep Q-Learning from Demonstrations (DQfD)	
    - Recurrent Replay Distributed DQN from Demonstratinos (R2D3)	
- DeepMind MuJoCo Multi-Agent Soccer Environment
    https://github.com/deepmind/dm_control/blob/master/dm_control/locomotion/soccer/README.md
- see acme code: https://github.com/deepmind/acme
"""


def train():

    # Start a gRPC server at port 6009
    if Params.TENSORFLOW_PROFILER:
        tf.profiler.experimental.server.start(6009)

    # Set TF dtype and filter devices
    tf.keras.backend.set_floatx(Params.DTYPE)
    # environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Set random seed
    if Params.SEED is not None:
        environ['PYTHONHASHSEED'] = str(Params.SEED)
        tf.random.set_seed(Params.SEED)

    # Init Logger
    logger = Logger()

    # Init Video Recorder
    if Params.RECORD_VIDEO:
        video_recorder = VideoRecorder()
        # record_episode = tf.Variable(Params.RECORD_VIDEO)
    else:
        video_recorder = None

    # Init replay buffer
    if Params.BUFFER_TYPE == "Uniform":
        replay_buffer = UniformReplayBuffer()
    elif Params.BUFFER_TYPE == "ReverbUniform":
        replay_buffer = ReverbUniformReplayBuffer()
    elif Params.BUFFER_TYPE == "ReverbPrioritized":
        replay_buffer = ReverbPrioritizedReplayBuffer()
    else:
        raise Exception(f"Buffer with name {Params.BUFFER_TYPE} not found.")

    # Init Actor-Noise
    if Params.NOISE_TYPE == "Gaussian":
        actor_noise = GaussianNoise()
    elif Params.NOISE_TYPE == "OrnsteinUhlenbeck":
        actor_noise = TFOrnsteinUhlenbeckActionNoise()
    else:
        raise Exception(f"Noise with name {Params.NOISE_TYPE} not found.")

    # Create threading event
    actor_event_stop = threading.Event()

    ## Init learner
    learner = Learner(logger, replay_buffer)
    learner_thread = threading.Thread(target=learner.run)
    learner_thread.start()

    ## Init actors
    threads = []
    for n_actor in range(Params.NUM_ACTORS):
        actor = Actor(n_actor, actor_event_stop, learner.policy_variables,
                      replay_buffer, actor_noise, logger, video_recorder)
        thread = threading.Thread(target=actor.run)
        thread.start()
        threads.append(thread)

    ## Wait for learner to finish
    learner_thread.join()

    ## Stop actors and wait for finish
    actor_event_stop.set()
    for t in threads:
        t.join()

    # Stop reverb server
    if Params.BUFFER_FROM_REVERB:
        replay_buffer.reverb_server.stop()


if __name__ == '__main__':
    train()



