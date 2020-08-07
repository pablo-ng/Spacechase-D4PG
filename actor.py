import tensorflow as tf

from params import Params
from utils import tf_round
from networks import ActorNetwork
from game_tf import GameTF
from gym_wrapper_tf import GymTF


class Actor(ActorNetwork):

    def __init__(self, actor_id, actor_event_run, actor_event_stop, replay_buffer, actor_noise):
        super(Actor, self).__init__(with_target_net=False, name="Actor")

        with tf.device(self.device), self.name_scope:

            self.actor_id = actor_id
            self.actor_event_run = actor_event_run
            self.actor_event_stop = actor_event_stop
            self.replay_buffer = replay_buffer
            self.actor_noise = actor_noise

            ## Init Env
            if Params.ENV_NAME == "SC":
                self.env = GameTF()
            elif Params.ENV_NAME == "GYM":
                self.env = GymTF()
            else:
                raise Exception(f"Environment with name {Params.ENV_NAME} not found.")

    def run(self):
        print("retracing actor run")

        with tf.device(self.device), self.name_scope:

            ## Set threading event
            self.actor_event_run.set()

            tf.while_loop(
                lambda _: tf.logical_not(self.actor_event_stop.is_set()),
                self.do_episode,
                loop_vars=[tf.constant(0)]
            )

    def do_episode(self, n_episode):
        print("retracing do_episode")

        with tf.device(self.device), self.name_scope:

            # Update episode number
            n_episode = tf.add(n_episode, 1)

            # Set record state
            # record_episode.assign(tf.cond(
            #     tf.logical_and(
            #         tf.logical_and(Params.RECORD_VIDEO,
            #                        tf.greater_equal(n_episode, tf.constant(Params.RECORD_START_EP))),
            #         tf.equal(tf.math.floormod(n_episode, Params.RECORD_FREQ), 0)
            #     ), lambda: True, lambda: False))

            # Get env initial state as shape (1, space)
            state0 = self.env.reset()

            # Do ep steps
            ep_steps, _, _, ep_reward, ep_avg_q_max, frames = tf.while_loop(
                lambda *args: tf.math.logical_and(
                    args[0] < Params.MAX_EP_STEPS,
                    tf.math.logical_not(args[1])
                ),
                self.do_step,
                loop_vars=[0, False, state0, 0., 0., tf.TensorArray(tf.uint8, size=1, dynamic_size=True)]
            )

            ## Decrease actor noise sigma after episode
            tf.cond(tf.less(self.env.n_steps_total, Params.WARM_UP_STEPS), lambda: None, self.actor_noise.decrease_sigma)

            # Update return values
            ep_avg_q_max = tf.divide(ep_avg_q_max, tf.cast(ep_steps, Params.DTYPE))
            ep_avg_reward = ep_reward / tf.cast(ep_steps, Params.DTYPE)

            # Log episode
            tf.print('E:', n_episode, '\tSteps:', ep_steps, '\t\tAvg. Reward:', tf_round(ep_avg_reward, 3),
                     '\tAvg. Qmax:',
                     tf_round(ep_avg_q_max), '\tNoise variance:', tf_round(self.actor_noise.sigma, 2), '\t', self.env.info)

            # Save video if recorded
            # ep_replay_filename = tf.cond(record_episode, lambda: tf.py_function(recorder.save_video,
            #                                                                     inp=[frames.stack(), n_episode,
            #                                                                          tf.round(ep_avg_reward)],
            #                                                                     Tout=tf.string), lambda: "")

            # Log tensorboard
            # tf.cond(Params.LOG_TENSORBOARD,
            #         lambda: log_ep_tensorboard(n_episode, ep_steps, ep_avg_reward, ep_avg_q_max, ep_replay_filename),
            #         tf.no_op)

            # Update actor network with learner params
            if n_episode % Params.UPDATE_ACTOR_FREQ == 0:
                pass
                # todo

            return n_episode

    # noinspection PyUnusedLocal
    def do_step(self, j, terminal, state, ep_reward_sum, ep_q_max_sum, frames):
        print("retracing do_step")

        with tf.device(self.device), self.name_scope:

            # Predict next action / random in warm up phase, adding exploration noise, returns (1, act_space)
            action = tf.reshape(
                tf.cond(
                    tf.less(self.env.n_steps_total, Params.WARM_UP_STEPS),
                    lambda: tf.random.uniform((Params.ENV_ACT_SPACE.dims[0],), minval=-Params.ENV_ACT_BOUND, maxval=Params.ENV_ACT_BOUND),
                    lambda: tf.add(self.predict_action(tf.reshape(state, (1, -1))), self.actor_noise)
                ), (self.env.act_space.dims[0],)
            )

            # Save action
            # ep_actions = ep_actions.write(j, action[0, 0])

            # Perform step in env
            state2, reward, terminal = self.env.step(action)

            # Save next frame if recording
            # frames = tf.cond(tf.logical_and(record_episode, tf.equal(tf.math.floormod(j, Params.RECORD_STEP_FREQ), 0)),
            #                  lambda: frames.write(frames.size(), env.get_frame()), lambda: frames)

            max_q = 0.
            # todo maxq

            # Wait for learner ?
            # self.actor_event_run.wait()

            # Add to replay memory as shape (1, space)
            self.replay_buffer.append((state, action, reward, terminal, state2))

            return tf.add(j, 1), terminal, state2, tf.add(ep_reward_sum, reward), tf.add(ep_q_max_sum, max_q), frames


