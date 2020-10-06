import tensorflow as tf
import gym
from params import Params
gym.logger.set_level(40)


class GymTF(tf.Module):

    def __init__(self):
        super(GymTF, self).__init__(name="GymTF")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            self.dtype = Params.DTYPE
            self.env = gym.make('Pendulum-v0')
            self.frame_size = Params.FRAME_SIZE

            self.obs_space = tf.TensorShape(self.env.observation_space.shape[0],)
            self.act_space = tf.TensorShape(self.env.action_space.shape[0],)
            self.act_bound = tf.constant(self.env.action_space.high)
            assert (self.env.action_space.high == -self.env.action_space.low)

            self.n_steps_total = tf.Variable(0)
            self.info = tf.constant("")

            if Params.SEED is not None:
                self.env.seed(Params.SEED)

            self.reset()

    def reset(self):
        print("retracing GymTF reset")
        with tf.device(self.device), self.name_scope:
            state = tf.py_function(func=self.env_reset, inp=[], Tout=[self.dtype])

            state = tf.convert_to_tensor(state, dtype=self.dtype)
            return tf.reshape(state, (-1,))

    def env_reset(self):
        state = self.env.reset()
        # self.env.state = state
        return state

    def step(self, action):
        print("retracing GymTF step")
        with tf.device(self.device), self.name_scope:
            self.n_steps_total.assign_add(1)

            state2, reward, terminal = tf.py_function(func=self.env_step, inp=[action], Tout=[self.dtype, self.dtype, tf.bool])

            state2 = tf.reshape(state2, (-1,))
            reward = tf.reshape(reward, ())
            terminal = tf.reshape(terminal, ())

            return state2, reward, terminal

    def env_step(self, action):
        state2, reward, terminal, _ = self.env.step(action)
        return state2, reward, terminal

    def warmup_action(self):
        return tf.random.uniform((self.act_space.dims[0],), minval=-self.act_bound, maxval=self.act_bound)

    def get_frame(self):
        with tf.device(self.device), self.name_scope:
            return tf.cast(tf.zeros(tf.cast(self.frame_size, tf.int32)), tf.uint8)



