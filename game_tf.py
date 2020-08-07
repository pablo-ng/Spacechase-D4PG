import tensorflow as tf
from params import Params


class GameTF(tf.Module):

    def __init__(self):

        super(GameTF, self).__init__(name="GameTF")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            self.dtype = Params.DTYPE
            self.dt = Params.DT

            self.frame_size = Params.FRAME_SIZE

            self.bes = tf.Variable([0., 0.])
            self.ges = tf.Variable([0., 0.])
            self.pos = tf.Variable([0., 0.])
            self.rad = tf.constant(0.01)
            self.bes_inc = tf.constant(0.2)

            self.n_goals = tf.constant(4)
            self.goal_pos = tf.Variable(tf.zeros((self.n_goals, 2)))

            self.n_steps_total = tf.Variable(0)
            self.terminal = tf.Variable(True)
            self.info = tf.Variable("")

            self.obs_space = tf.TensorShape(4 + 2*self.n_goals,)
            self.act_space = tf.TensorShape(2,)
            self.act_bound = tf.constant([1.])

            self.not_schnitt_kreis_bande = \
                lambda kreis: tf.cond(
                    tf.logical_or(
                        tf.logical_or(
                            (kreis[0] - kreis[2]) <= 0, (kreis[1] - kreis[2]) <= 0
                        ),
                        tf.logical_or(
                            (kreis[0] + kreis[2]) >= 1, (kreis[1] + kreis[2]) >= 1
                        ),
                    ), lambda: False, lambda: True
                )

            self.schnitt_kreis_bande = lambda kreis: tf.math.logical_not(self.not_schnitt_kreis_bande(kreis))

    def reset(self):
        print("retracing reset")

        with tf.device(self.device), self.name_scope:

            self.terminal.assign(False)
            self.info.assign("")
            self.ges.assign([0., 0.])
            self.pos.assign([0.9, 0.1])

            x = tf.transpose(tf.random.uniform((1, tf.math.floordiv(self.n_goals, 2)), minval=0.2, maxval=0.4))
            y = tf.transpose(tf.random.uniform((1, tf.math.floordiv(self.n_goals, 2)), minval=0.1, maxval=0.8))
            self.goal_pos.assign(tf.concat([tf.concat([y, x], axis=1), tf.concat([y, 1-x], axis=1)], axis=0))

            return self.get_obs()

    def step(self, action):

        print("retracing step")

        with tf.device(self.device), self.name_scope:

            ## Zeit
            self.n_steps_total.assign_add(1)

            ## Movement
            self.bes.assign(self.bes_inc * action)
            self.ges.assign_add(self.bes * self.dt)
            self.pos.assign_add(self.ges * self.dt)

            ## Banden
            kreis_rot = tf.stack([self.pos[0], self.pos[1], self.rad])
            rot_trifft_bande = tf.cond(self.schnitt_kreis_bande(kreis_rot), lambda: True, lambda: False)

            ## Auswertung
            self.terminal.assign(tf.cond(
                rot_trifft_bande,
                lambda: True, lambda: False
            ))

            ## Calc reward and remove goals
            distances = tf.map_fn(tf.norm, tf.subtract(self.pos, self.goal_pos))
            reached_goal = tf.cond(
                tf.less_equal(tf.reduce_min(distances), .1),
                lambda: True, lambda: False
            )
            reward = tf.cond(
                reached_goal,
                lambda: 1., lambda: 0.
            )
            self.goal_pos = tf.cond(
                reached_goal,
                lambda: self.goal_pos.scatter_update(tf.IndexedSlices(tf.constant([-1., -1.]), tf.argmin(distances))),
                lambda: self.goal_pos
            )

            return self.get_obs(), reward, self.terminal

    def get_obs(self):
        print("retracing get_obs")
        with tf.device(self.device), self.name_scope:
            obs = tf.concat([self.pos, self.ges, tf.reshape(self.goal_pos, (-1,))], axis=0)
            obs = tf.expand_dims(obs, axis=0)
            return obs

    def get_frame(self):
        print("retracing get_frame")

        with tf.device(self.device), self.name_scope:

            indices = tf.expand_dims(tf.cast(self.pos * self.frame_size, tf.int64), axis=0)
            values = tf.constant([0.2])

            goal_pos_valid_indices = tf.reduce_any(tf.greater_equal(self.goal_pos, 0), axis=1)
            goal_pos_filtered = self.goal_pos[goal_pos_valid_indices]
            indices = tf.concat([indices, tf.cast(goal_pos_filtered * self.frame_size, tf.int64)], axis=0)
            values = tf.concat([values, tf.repeat(.6, repeats=tf.reduce_sum(tf.cast(goal_pos_valid_indices, tf.int64)))], axis=0)

            frame = tf.SparseTensor(indices=indices, values=values, dense_shape=tf.cast(self.frame_size + 1, tf.int64))
            frame = tf.sparse.to_dense(frame, validate_indices=False)
            frame = tf.expand_dims(frame, axis=0)
            frame = tf.expand_dims(frame, axis=3)

            filters = tf.ones((3, 3, 1, 1))
            frame = tf.nn.conv2d(frame, filters, strides=1, padding="SAME")

            frame = tf.squeeze(frame)
            frame = tf.cast(frame * 255, tf.uint8)

            return frame




############################################################################## OLD GAME TF:

# import tensorflow as tf
#
#
#
# class Player:
#
#     # instantiate shared class variables here!
#
#     def __init__(self, rad):
#         print("retracing player init")
#         self.reward = tf.Variable(0.)
#         self.rad = rad
#         self.pos = tf.Variable([0., 0.])
#         self.ges = tf.Variable([0., 0.])
#         self.bes = tf.Variable([0., 0.])
#
#     # @tf.function(input_signature=())
#     def reset(self):
#         print("retracing player reset")
#         self.reward.assign(0.)
#         self.pos.assign([0., 0.])
#         self.ges.assign([0., 0.])
#         self.bes.assign([0., 0.])
#
#
# class GameTF:
#
#     def __init__(self, dtype, frame_size, dt):
#
#         print("retracing init")
#
#         self.dtype = dtype
#         self.dt = dt
#         self.t_end = tf.constant(60.)
#         self.n_t = tf.round(self.t_end / self.dt)
#         self.bes = tf.constant(0.1)
#         self.pi = tf.constant(3.14159265359)
#         self.eps = tf.constant(1e-8)
#
#         self.obs_space = tf.TensorShape(3,)
#         self.act_space = tf.TensorShape(2,)
#         self.act_bound = tf.constant([1.])
#
#         self.spaceball_radius = tf.constant(0.01)
#
#         self.reward_survival = tf.constant(+0.1)
#         self.reward_failing = tf.constant(-10.)
#
#         self.frame_size = frame_size
#
#         self.goal_pos = tf.Variable([0., 0.])
#         # self.last_distance = tf.Variable(0.)
#         self.n_steps_total = tf.Variable(0)
#         self.i_t = tf.Variable(0.)
#         self.t = tf.Variable(0.)
#         self.running = tf.Variable(False)
#         self.info = tf.Variable("")
#
#         self.rot = Player(self.spaceball_radius)
#
#         self.not_schnitt_kreis_bande = \
#             lambda kreis: tf.cond(
#                 tf.logical_or(
#                     tf.logical_or(
#                         (kreis[0] - kreis[2]) <= 0, (kreis[1] - kreis[2]) <= 0
#                     ),
#                     tf.logical_or(
#                         (kreis[0] + kreis[2]) >= 1, (kreis[1] + kreis[2]) >= 1
#                     ),
#                 ), lambda: tf.constant(False), lambda: tf.constant(True)
#             )
#
#         self.schnitt_kreis_bande = lambda kreis: tf.math.logical_not(self.not_schnitt_kreis_bande(kreis))
#
#     def reset(self):
#
#         print("retracing reset")
#
#         # self.last_distance.assign(10.)
#         self.goal_pos.assign(tf.random.uniform((2,), minval=0.25, maxval=0.75))
#         self.i_t.assign(0)
#         self.t.assign(0)
#         self.running.assign(True)
#         self.info.assign("")
#
#         self.rot.reset()
#         self.rot.ges.assign([0.2, 0.2])
#
#         # tf.random.uniform((2,), minval=0.25, maxval=0.75)  tf.fill((2,), 0.5)
#         # todo change to: np.array([1.5 * self.spaceball_radius, 1.5 * self.spaceball_radius]) # tf.random.uniform(()), tf.random.uniform(())
#         self.rot.pos.assign([0.5, 0.5])
#
#         return self.get_obs()
#
#     def step(self, action):
#
#         action = tf.squeeze(action, axis=0)
#         tf.cond(tf.reduce_sum(tf.abs(action)) > 5, lambda: tf.print("Action:", action), lambda: tf.no_op())
#         print("retracing step")
#
#         ## Zeit
#         self.t.assign(self.i_t * self.dt)
#         self.i_t.assign_add(1)
#         self.n_steps_total.assign_add(1)
#         self.running.assign(tf.cond(self.i_t >= self.n_t, lambda: tf.constant(False), lambda: tf.constant(True)))
#         self.info.assign(tf.add(self.info, tf.cond(self.running, lambda: "", lambda: "Zeit ist abgelaufen.")))
#
#         ## Movement
#         dir_sign = tf.math.sign(action[1])
#         angle = tf.multiply(action[0], self.pi / 2)  # pi/2 = 90° = -1 left .. +1 right
#         direction = self.rot.ges
#
#         ges_rotated = dir_sign * tf.squeeze([
#             tf.math.cos(angle) * direction[0] - tf.math.sin(angle) * direction[1],
#             tf.math.sin(angle) * direction[0] + tf.math.cos(angle) * direction[1],
#         ])
#         self.rot.ges.assign(ges_rotated)
#         self.rot.pos.assign_add(self.rot.ges * self.dt)
#
#         ## Banden
#         kreis_rot = tf.stack([self.rot.pos[0], self.rot.pos[1], self.rot.rad])
#         rot_trifft_bande = tf.cond(self.schnitt_kreis_bande(kreis_rot), lambda: tf.constant(True), lambda: tf.constant(False))
#
#         ## Auswertung
#         self.running.assign(tf.cond(
#             rot_trifft_bande,
#             lambda: tf.constant(False), lambda: self.running
#         ))
#
#         # Calc reward
#         distance = tf.norm(tf.subtract(self.rot.pos, self.goal_pos))
#         reward = tf.cond(distance < tf.constant(0.08), lambda: tf.constant(1.), lambda: tf.constant(0.01))
#         tf.cond(distance < tf.constant(0.08), lambda: self.goal_pos.assign(tf.random.uniform((2,), minval=0.25, maxval=0.75)), lambda: 1.)
#         # reward = 0.1 / tf.add(tf.norm(tf.subtract(self.rot.pos, self.goal_pos)), 0.1)
#         # reward = tf.cond(distance < self.last_distance, lambda: tf.constant(0.1), lambda: tf.constant(-0.2))
#         # self.last_distance.assign(distance)
#         self.rot.reward.assign(reward)
#
#         return self.get_obs(), self.rot.reward,  tf.logical_not(self.running)
#
#
#     def get_obs(self):
#         distance = tf.subtract(self.goal_pos, self.rot.pos)
#         angle = tf.math.atan2(distance[0], distance[1]) - tf.math.atan2(self.rot.ges[0], self.rot.ges[1])
#         angle = tf.divide(angle, 2*self.pi)
#         obs = tf.stack([angle, -angle, tf.norm(distance) / 1.41])  # tf.norm(distance) / 1.41, tf.norm(self.rot.ges)
#         obs = tf.expand_dims(obs, axis=0)
#
#
#         return obs
#         # obs_old = tf.expand_dims(tf.concat([self.rot.pos, self.rot.ges], axis=0), axis=0)
#         # a = tf.math.atan2(self.rot.ges[0], self.rot.ges[1])
#         # obs = tf.stack([self.rot.pos[0], self.rot.pos[1], a, -a, tf.norm(self.rot.bes)])
#         # obs = tf.expand_dims(obs, axis=0)
#         # return obs
#
#
#     # @tf.function(input_signature=[])
#     def get_frame(self):
#
#         print("retracing get_frame")
#
#         indices = []
#         values = []
#
#         # for idx in range(self.n_tanke):
#         #     indices.append(tf.cast(tf.gather(self.tanken, idx)[0:2] * self.frame_size, tf.int32))
#         #     values.append(tf.constant(0.6))
#
#         indices.append(tf.cast(self.rot.pos * self.frame_size, tf.int32))
#         values.append(tf.constant(0.2))
#
#         indices.append(tf.cast(self.goal_pos * self.frame_size, tf.int32))
#         values.append(tf.constant(0.6))
#
#         frame = tf.SparseTensor(indices=indices, values=values, dense_shape=tf.cast(self.frame_size + 1, tf.int64))
#         frame = tf.sparse.to_dense(frame, validate_indices=False)
#         frame = tf.expand_dims(frame, axis=0)
#         frame = tf.expand_dims(frame, axis=3)
#
#         filters = tf.ones((3, 3, 1, 1))
#         frame = tf.nn.conv2d(frame, filters, strides=1, padding="SAME")
#
#         frame = tf.squeeze(frame)
#         frame = tf.cast(frame * 255, tf.uint8)
#         return frame
#
#
