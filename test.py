import tensorflow as tf
import datetime

from params import Params, ParamsTest
from video_recorder import VideoRecorder
from logger import Logger

from gym_wrapper_tf import GymTF
from game_tf import GameTF


def test():

    # Set TF / Keras dtype
    tf.keras.backend.set_floatx(Params.DTYPE)

    # Load model
    model = tf.keras.models.load_model(ParamsTest.MODEL_PATH, compile=False)

    # Construct logdir
    log_dir = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_test"

    # Init Logger
    if Params.DO_LOGGING:
        logger = Logger("logs/" + log_dir)
    else:
        logger = None

    # Init Video Recorder
    if Params.RECORD_VIDEO:
        video_recorder = VideoRecorder("recorded/" + log_dir)
    else:
        video_recorder = None

    # Init Env
    env = eval(Params.ENV_NAME)()

    run(logger, env, model, video_recorder)


@tf.function()
def run(logger, env, model, video_recorder):
    tf.while_loop(
        lambda n_episode: tf.less(n_episode, ParamsTest.N_EPISODES),
        lambda *args: do_episode(logger, env, model, video_recorder, *args),
        loop_vars=[tf.constant(0)]
    )


def do_episode(logger, env, model, video_recorder, n_episode):

    # Get env initial state as shape (1, space)
    state0 = env.reset()

    # Do ep steps
    ep_steps, _, _, ep_reward_sum, ep_reward_sum_discounted, ep_frames = tf.while_loop(
        lambda *args: args[1],
        lambda *args: do_step(env, model, *args),
        loop_vars=[
            tf.constant(0), tf.constant(True), state0, tf.constant(0.), tf.constant(0.),
            tf.TensorArray(tf.uint8, size=1, dynamic_size=True),
        ]
    )

    # Compute average reward
    ep_avg_reward = ep_reward_sum / tf.cast(ep_steps, Params.DTYPE)

    # Save video
    ep_replay_filename = tf.py_function(video_recorder.save_video, inp=[ep_frames.stack(), n_episode], Tout=tf.string)

    # Log episode
    logger.actor_steps_counter.increment(ep_steps)
    logger.log_ep_actor(n_episode, ep_steps, ep_avg_reward, ep_reward_sum_discounted,
                        tf.constant(0.), env.info, ep_replay_filename),

    return tf.add(n_episode, 1)


# noinspection PyUnusedLocal
def do_step(env, model, n_step, terminal, state, ep_reward_sum, ep_reward_sum_discounted, frames):

    # Predict next action
    action = tf.reshape(model(tf.expand_dims(state, axis=0)), (Params.ENV_ACT_SPACE.dims[0],))

    # Perform step in env
    state2, reward, terminal = env.step(action)

    # Save next frame
    frames = tf.cond(
        tf.equal(tf.math.floormod(n_step, Params.RECORD_STEP_FREQ), 0),
        lambda: frames.write(frames.size(), env.get_frame(as_image=True)),
        lambda: frames
    )

    # Increase step counter
    n_step = tf.add(n_step, 1)
    continue_episode = tf.math.logical_and(
        n_step < Params.MAX_EP_STEPS,
        tf.math.logical_not(terminal)
    )

    return n_step, continue_episode, state2, tf.add(ep_reward_sum, reward), \
        tf.add(ep_reward_sum_discounted, tf.pow(Params.GAMMA, tf.cast(n_step-1, Params.DTYPE)) * reward), frames,


if __name__ == '__main__':
    test()


