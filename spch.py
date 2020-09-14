import tensorflow as tf
from os import environ
import threading

from params import Params
from actor import Actor
from learner import Learner

"""
TODO
- name all operations, use profiler: 
    https://www.tensorflow.org/guide/profiler#profiling_apis
    https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
- Learning from demonstrations
    - Deep Q-Learning from Demonstrations (DQfD)	
    - Recurrent Replay Distributed DQN from Demonstratinos (R2D3)	
- DeepMind MuJoCo Multi-Agent Soccer Environment
    https://github.com/deepmind/dm_control/blob/master/dm_control/locomotion/soccer/README.md
- see acme code: https://github.com/deepmind/acme
- see matlab params: https://de.mathworks.com/products/reinforcement-learning.html
"""


def train():

    # Start a gRPC server at port 6009
    if Params.TENSORFLOW_PROFILER:
        tf.profiler.experimental.server.start(6009)

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

    ## Init threads list
    threads = []

    ## Create threading event
    actor_event_stop = threading.Event()

    ## Init learner
    learner = Learner(actor_event_stop)
    thread = threading.Thread(target=learner.run)
    thread.start()
    threads.append(thread)

    ## Init actors
    for n_actor in range(Params.NUM_ACTORS):
        actor = Actor(n_actor, actor_event_stop, learner.policy_variables)
        thread = threading.Thread(target=actor.run)
        thread.start()
        threads.append(thread)

    ## Wait for finish
    for t in threads:
        t.join()


if __name__ == '__main__':
    train()



