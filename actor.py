import tensorflow as tf

from params import Params
from networks import ActorNetwork, update_weights
from gym_wrapper_tf import GymTF
from game_tf import GameTF


class Actor(ActorNetwork):

    def __init__(self, actor_id, learner_policy_variables, replay_buffer, actor_noise, logger, video_recorder):
        super(Actor, self).__init__(with_target_net=False, name="Actor")

        with tf.device(self.device), self.name_scope:

            self.n_step_returns = Params.N_STEP_RETURNS
            self.gamma = Params.GAMMA

            self.actor_id = actor_id
            self.learner_policy_variables = learner_policy_variables
            self.replay_buffer = replay_buffer.get_client() if Params.BUFFER_FROM_REVERB else replay_buffer
            self.actor_noise = actor_noise
            self.logger = logger
            self.video_recorder = video_recorder

            self.record_episode = tf.Variable(False)
            self.n_episode = tf.Variable(0)
            self.running = tf.Variable(True)

            # Init Env
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
                lambda: self.running,
                self.do_episode,
                loop_vars=[]
            )

    def do_episode(self):
        print("retracing do_episode")

        with tf.device(self.device), self.name_scope:

            # Get episode number
            self.n_episode.assign(self.logger.increment_episode())

            # Set record state
            self.record_episode.assign(tf.cond(
                tf.logical_and(
                    tf.logical_and(Params.RECORD_VIDEO,
                                   tf.greater_equal(self.n_episode, Params.RECORD_START_EP)),
                    tf.equal(tf.math.floormod(self.n_episode - Params.RECORD_START_EP, Params.RECORD_FREQ), 0)
                ), lambda: True, lambda: False))

            # Get env initial state as shape (1, space)
            state0 = self.env.reset()

            # Do ep steps
            ep_steps, _, _, ep_reward_sum, ep_reward_sum_discounted, ep_frames, _, _, _ = tf.while_loop(
                lambda *args: args[1],
                self.do_step,
                loop_vars=[
                    tf.constant(0), tf.constant(True), state0, tf.constant(0.), tf.constant(0.),
                    tf.TensorArray(tf.uint8, size=1, dynamic_size=True),
                    tf.TensorArray(Params.DTYPE, size=self.n_step_returns, dynamic_size=False,
                                   element_shape=Params.ENV_OBS_SPACE),
                    tf.TensorArray(Params.DTYPE, size=self.n_step_returns, dynamic_size=False,
                                   element_shape=Params.ENV_ACT_SPACE),
                    tf.TensorArray(Params.DTYPE, size=self.n_step_returns, dynamic_size=False,
                                   element_shape=()),
                ]
            )

            # Decrease actor noise sigma after episode
            tf.cond(tf.less(self.env.n_steps_total, Params.WARM_UP_STEPS),
                    lambda: None,
                    self.actor_noise.decrease_sigma)

            # Compute average reward
            ep_avg_reward = ep_reward_sum / tf.cast(ep_steps, Params.DTYPE)

            # Save video if recorded
            ep_replay_filename = tf.cond(
                self.record_episode,
                lambda: tf.py_function(self.video_recorder.save_video,
                                       inp=[ep_frames.stack(), self.n_episode],
                                       Tout=tf.string),
                lambda: ""
            )

            # Log episode
            tf.cond(
                tf.equal(tf.math.mod(self.n_episode, tf.constant(Params.ACTOR_LOG_STEPS)), tf.constant(0)),
                lambda: self.logger.log_ep_actor(self.n_episode, ep_steps, ep_avg_reward, ep_reward_sum_discounted,
                                                 self.actor_noise.sigma, self.env.info, ep_replay_filename),
                lambda: None, name="Logger"
            )

            # Update actor network with learner params
            tf.cond(
                tf.equal(tf.math.mod(self.n_episode, Params.UPDATE_ACTOR_FREQ), 0),
                lambda: update_weights(self.tvariables + self.nvariables,
                                       self.learner_policy_variables, tf.constant(1.)),
                tf.no_op
            )

            return []

    # noinspection PyUnusedLocal
    def do_step(self, n_step, terminal, state, ep_reward_sum, ep_reward_sum_discounted,
                frames, states_buffer, actions_buffer, rewards_buffer):
        print("retracing do_step")

        with tf.device(self.device), self.name_scope:

            # Predict next action / random in warm up phase, adding exploration noise, returns (1, act_space)
            action = tf.cond(
                tf.less(self.env.n_steps_total, Params.WARM_UP_STEPS),
                lambda: self.env.warmup_action(),
                lambda: tf.reshape(
                    tf.add(
                        self.predict_action(tf.expand_dims(state, axis=0)),
                        self.actor_noise.__call__()
                    ), (Params.ENV_ACT_SPACE.dims[0],)
                ),
            )

            # Perform step in env
            state2, reward, terminal = self.env.step(action)

            # Print warm up info
            tf.cond(
                tf.equal(self.env.n_steps_total, Params.WARM_UP_STEPS),
                lambda: tf.print(f"Actor {self.actor_id} finish warm up at episode", self.n_episode, "/ step", n_step),
                lambda: tf.constant(False),
            )

            # Save next frame if recording
            frames = tf.cond(
                tf.logical_and(self.record_episode, tf.equal(tf.math.floormod(n_step, Params.RECORD_STEP_FREQ), 0)),
                lambda: frames.write(frames.size(), self.env.get_frame()),
                lambda: frames
            )

            # Increase step counter
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

                # Add to replay memory
                if Params.BUFFER_FROM_REVERB:
                    _rewards_buffer_stack = tf.concat(
                        (tf.repeat(-Params.ENV_REWARD_INF, i), rewards_buffer_stack[i:]),
                        axis=0)
                    # _rewards_buffer_stack = rewards_buffer_stack
                    if Params.BUFFER_IS_PRIORITIZED:
                        max_priority = self.replay_buffer.sample(Params.BUFFER_TYPE + "_max",
                                                                 Params.BUFFER_DATA_SPEC_DTYPES).info.priority
                        self.replay_buffer.insert(
                            [_state0, action0, _rewards_buffer_stack, terminal, state2, tf.zeros(Params.NUM_ATOMS)],
                            tables=Params.BUFFER_PRIORITY_TABLE_NAMES,
                            priorities=tf.repeat(max_priority, Params.BUFFER_PRIORITY_TABLE_NAMES.shape[0])
                        )
                    else:
                        self.replay_buffer.insert(
                            [_state0, action0, _rewards_buffer_stack, terminal, state2, tf.zeros(Params.NUM_ATOMS)],
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
                tf.add(ep_reward_sum_discounted, tf.pow(self.gamma, tf.cast(n_step-1, Params.DTYPE)) * reward), \
                frames, states_buffer, actions_buffer, rewards_buffer





