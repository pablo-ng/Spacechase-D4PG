import tensorflow as tf

from params import Params
from networks import ActorNetwork, update_weights
from gym_wrapper_tf import GymTF
from game_tf import GameTF


class Actor(ActorNetwork):

    def __init__(self, actor_id, actor_event_stop, learner_policy_variables,
                 replay_buffer, actor_noise, logger, video_recorder):
        super(Actor, self).__init__(with_target_net=False, name="Actor")

        with tf.device(self.device), self.name_scope:

            self.actor_id = actor_id
            self.actor_event_stop = actor_event_stop
            self.update_op = tf.no_op
            self.n_step_returns = Params.N_STEP_RETURNS
            self.gamma = Params.GAMMA
            self.learner_policy_variables = learner_policy_variables
            self.indices = tf.range(self.n_step_returns)
            self.replay_buffer = replay_buffer.get_client() if Params.BUFFER_FROM_REVERB else replay_buffer
            self.actor_noise = actor_noise
            self.video_recorder = video_recorder
            self.logger = logger

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
            ep_steps, _, _, ep_reward, ep_frames, _, _, _ = tf.while_loop(
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

            self.logger.increment_episode()

            ## Decrease actor noise sigma after episode
            tf.cond(tf.less(self.env.n_steps_total, Params.WARM_UP_STEPS), lambda: None, self.actor_noise.decrease_sigma)

            ## Update return values
            ep_avg_reward = ep_reward / tf.cast(ep_steps, Params.DTYPE)

            # Save video if recorded
            ep_replay_filename = ""
            # ep_replay_filename = tf.cond(record_episode, lambda: tf.py_function(recorder.save_video,
            #                                                                     inp=[ep_frames.stack(), n_episode,
            #                                                                          tf.round(ep_avg_reward)],
            #                                                                     Tout=tf.string), lambda: "")

            ## Log episode
            self.logger.log_ep_actor(ep_steps, ep_avg_reward, self.actor_noise.sigma, self.env.info, ep_replay_filename)

            ## Update actor network with learner params
            tf.cond(
                tf.equal(tf.math.mod(self.logger.n_episode(), Params.UPDATE_ACTOR_FREQ), 0),
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
                    lambda: tf.random.uniform((Params.ENV_ACT_SPACE.dims[0],),
                                              minval=-Params.ENV_ACT_BOUND,
                                              maxval=Params.ENV_ACT_BOUND),
                    lambda: tf.add(self.predict_action(tf.reshape(state, (1, -1))),
                                   self.actor_noise.__call__())
                ), (self.env.act_space.dims[0],)
            )

            ## Perform step in env
            state2, reward, terminal = self.env.step(action)

            ## Save next frame if recording
            # frames = tf.cond(tf.logical_and(record_episode, tf.equal(tf.math.floormod(n_step, Params.RECORD_STEP_FREQ), 0)),
            #                  lambda: frames.write(frames.size(), env.get_frame()), lambda: frames)

            ## Increase step counter
            n_step = tf.add(n_step, 1)
            continue_episode = tf.math.logical_and(
                n_step < Params.MAX_EP_STEPS,
                tf.math.logical_not(terminal)
            )

            # Append state, action, reward to buffers
            buffer_write_idx = tf.math.mod(tf.subtract(n_step, 1), self.n_step_returns)
            states_buffer = states_buffer.write(buffer_write_idx, state)
            actions_buffer = actions_buffer.write(buffer_write_idx, action)
            rewards_buffer = rewards_buffer.write(buffer_write_idx, reward)
            rewards_buffer_stack = rewards_buffer.stack()

            loop_len = tf.cond(
                tf.greater_equal(n_step, Params.N_STEP_RETURNS),
                lambda: tf.cond(continue_episode, lambda: tf.constant(1), lambda: self.n_step_returns),
                lambda: tf.constant(0)
            )

            def append_traces(i):

                buffer_idx = tf.math.mod(n_step + i, self.n_step_returns)

                _state0 = states_buffer.read(buffer_idx)
                action0 = actions_buffer.read(buffer_idx)

                ## Add to replay memory
                if Params.BUFFER_FROM_REVERB:
                    self.replay_buffer.insert(
                        [_state0, action0, rewards_buffer_stack, terminal, state2, tf.zeros(Params.NUM_ATOMS)],
                        tables=tf.constant([Params.BUFFER_TYPE]),
                        priorities=tf.constant([1.], dtype=tf.float64)
                    )
                else:
                    gammas = Params.GAMMAS2[(i + buffer_idx):(buffer_idx + Params.N_STEP_RETURNS)]
                    discounted_reward = tf.reduce_sum(tf.multiply(rewards_buffer_stack[i:], gammas))
                    self.replay_buffer.append((_state0, action0, discounted_reward, terminal, state2, gammas[-1]))

                return tf.add(i, 1)

            tf.while_loop(
                lambda i: tf.less(i, loop_len),
                append_traces,
                loop_vars=[tf.constant(0)]
            )

            return n_step, continue_episode, state2, tf.add(ep_reward_sum, reward), \
                frames, states_buffer, actions_buffer, rewards_buffer





