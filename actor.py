import tensorflow as tf

from params import Params
from utils import tf_round, Logger, GaussianNoise, TFOrnsteinUhlenbeckActionNoise, EpisodeCounter
from networks import ActorNetwork, update_weights
from game_tf import GameTF
from gym_wrapper_tf import GymTF
from experience_replay import ReplayBuffer


class Actor(ActorNetwork):

    def __init__(self, actor_id, actor_event_stop, learner_policy_variables):
        super(Actor, self).__init__(with_target_net=False, name="Actor")

        with tf.device(self.device), self.name_scope:

            self.actor_id = actor_id
            self.actor_event_stop = actor_event_stop
            self.update_op = tf.no_op
            self.n_step_returns = Params.N_STEP_RETURNS
            self.gamma = Params.GAMMA
            self.learner_policy_variables = learner_policy_variables
            self.indices = tf.range(self.n_step_returns)

            if Params.BUFFER_TYPE in ("ReverbUniform", "ReverbPrioritized"):
                self.replay_buffer = ReplayBuffer.get_replay_buffer().get_client()
            else:
                self.replay_buffer = ReplayBuffer.get_replay_buffer()

            ## Init Actor-Noise
            if Params.NOISE_TYPE == "Gaussian":
                self.actor_noise = GaussianNoise
            elif Params.NOISE_TYPE == "OrnsteinUhlenbeck":
                self.actor_noise = TFOrnsteinUhlenbeckActionNoise
            else:
                raise Exception(f"Noise with name {Params.NOISE_TYPE} not found.")

            ## Init Env
            if Params.ENV_NAME == "SC":
                self.env = GameTF()
            elif Params.ENV_NAME == "GYM":
                self.env = GymTF()
            else:
                raise Exception(f"Environment with name {Params.ENV_NAME} not found.")

    @tf.function()
    def run(self):
        print("retracing actor run")

        with tf.device(self.device), self.name_scope:

            tf.while_loop(
                lambda: tf.logical_not(tf.reshape(tf.py_function(self.actor_event_stop.is_set, inp=[], Tout=[tf.bool]), ())),
                self.do_episode,
                loop_vars=[]
            )

    def do_episode(self):
        print("retracing do_episode")

        with tf.device(self.device), self.name_scope:

            # Set record state
            # record_episode.assign(tf.cond(
            #     tf.logical_and(
            #         tf.logical_and(Params.RECORD_VIDEO,
            #                        tf.greater_equal(n_episode, tf.constant(Params.RECORD_START_EP))),
            #         tf.equal(tf.math.floormod(n_episode, Params.RECORD_FREQ), 0)
            #     ), lambda: True, lambda: False))

            ## Get env initial state as shape (1, space)
            state0 = self.env.reset()

            ## Do ep steps
            ep_steps, _, _, ep_reward, frames, _, _, _ = tf.while_loop(
                lambda *args: args[1],
                self.do_step,
                loop_vars=[
                    0, True, state0, 0.,
                    tf.TensorArray(tf.uint8, size=1, dynamic_size=True),
                    tf.TensorArray(Params.DTYPE, size=self.n_step_returns, dynamic_size=False, element_shape=Params.ENV_OBS_SPACE),
                    tf.TensorArray(Params.DTYPE, size=self.n_step_returns, dynamic_size=False, element_shape=Params.ENV_ACT_SPACE),
                    tf.TensorArray(Params.DTYPE, size=self.n_step_returns, dynamic_size=False, element_shape=()),
                ]
            )

            EpisodeCounter.increment()

            ## Decrease actor noise sigma after episode
            tf.cond(tf.less(self.env.n_steps_total, Params.WARM_UP_STEPS), lambda: None, self.actor_noise.decrease_sigma)

            ## Update return values
            ep_avg_reward = ep_reward / tf.cast(ep_steps, Params.DTYPE)

            # Save video if recorded
            ep_replay_filename = ""
            # ep_replay_filename = tf.cond(record_episode, lambda: tf.py_function(recorder.save_video,
            #                                                                     inp=[frames.stack(), n_episode,
            #                                                                          tf.round(ep_avg_reward)],
            #                                                                     Tout=tf.string), lambda: "")

            ## Log episode
            Logger.log_ep_actor(EpisodeCounter.__call__(), ep_steps, ep_avg_reward, self.actor_noise.sigma, self.env.info, ep_replay_filename)

            ## Update actor network with learner params
            tf.cond(
                tf.equal(tf.math.mod(EpisodeCounter.__call__(), Params.UPDATE_ACTOR_FREQ), 0),
                lambda: update_weights(self.tvariables + self.nvariables, self.learner_policy_variables, tf.constant(1.)),
                tf.no_op
            )

            return []

    # noinspection PyUnusedLocal
    def do_step(self, n_step, terminal, state, ep_reward_sum, frames, states_buffer, actions_buffer, rewards_buffer):
        print("retracing do_step")

        with tf.device(self.device), self.name_scope:

            ## Predict next action / random in warm up phase, adding exploration noise, returns (1, act_space)
            action = tf.reshape(
                tf.cond(
                    tf.less(self.env.n_steps_total, Params.WARM_UP_STEPS),
                    lambda: tf.random.uniform((Params.ENV_ACT_SPACE.dims[0],), minval=-Params.ENV_ACT_BOUND, maxval=Params.ENV_ACT_BOUND),
                    lambda: tf.add(self.predict_action(tf.reshape(state, (1, -1))), self.actor_noise.__call__())
                ), (self.env.act_space.dims[0],)
            )

            ## Perform step in env
            state2, reward, terminal = self.env.step(action)

            ## Save next frame if recording
            # frames = tf.cond(tf.logical_and(record_episode, tf.equal(tf.math.floormod(n_step, Params.RECORD_STEP_FREQ), 0)),
            #                  lambda: frames.write(frames.size(), env.get_frame()), lambda: frames)

            buffer_write_idx = tf.math.mod(n_step, self.n_step_returns)
            states_buffer = states_buffer.write(buffer_write_idx, state)
            actions_buffer = actions_buffer.write(buffer_write_idx, action)
            rewards_buffer = rewards_buffer.write(buffer_write_idx, reward)
            rewards_buffer_stack = rewards_buffer.stack()

            ## Increase step counter
            n_step = tf.add(n_step, 1)
            continue_episode = tf.math.logical_and(
                n_step < Params.MAX_EP_STEPS,
                tf.math.logical_not(terminal)
            )

            def compute_traces(i):

                buffer_idx = tf.math.mod(n_step + i, self.n_step_returns)

                state0 = states_buffer.gather(buffer_idx)
                action0 = actions_buffer.gather(buffer_idx)

                gammas = tf.vectorized_map(
                    lambda idx: tf.math.pow(self.gamma, tf.cast(tf.math.mod((buffer_idx + idx), self.n_step_returns), Params.DTYPE)),
                    self.indices[i:]
                )

                discounted_reward = tf.reduce_sum(tf.multiply(rewards_buffer_stack[i:], gammas))

                gamma = gammas[-1]

                ## Add to replay memory as shape (1, space)
                if Params.BUFFER_TYPE in ("ReverbUniform", "ReverbPrioritized"):
                    self.replay_buffer.insert(
                        (
                            tf.reshape(state0, (-1,)),
                            tf.reshape(action0, (-1,)),
                            tf.reshape(discounted_reward, (-1,)),
                            tf.reshape(terminal, (-1,)),
                            tf.reshape(state2, (-1,)),
                            tf.reshape(gamma, (-1,)),
                        ),
                        tables=tf.constant([Params.BUFFER_TYPE]),
                        priorities=tf.constant([1.], dtype=tf.float64)
                    )
                    # with self.replay_buffer.writer(max_sequence_length=1) as writer:
                    #     writer.append((state0, action0, discounted_reward, terminal, state2, gamma))
                    #     writer.create_item(table=Params.BUFFER_TYPE, num_timesteps=1, priority=1.)
                else:
                    self.replay_buffer.append((state0, action0, discounted_reward, terminal, state2, gamma))

                return tf.add(i, 1)

            loop_len = tf.cond(
                tf.greater_equal(n_step, Params.N_STEP_RETURNS),
                lambda: tf.cond(continue_episode, lambda: tf.constant(1), lambda: self.n_step_returns),
                lambda: tf.constant(0)
            )
            tf.while_loop(
                lambda i: tf.less(i, loop_len),
                compute_traces,
                loop_vars=[tf.constant(0)]
            )

            return n_step, continue_episode, state2, tf.add(ep_reward_sum, reward), frames, states_buffer, actions_buffer, rewards_buffer






