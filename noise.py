import tensorflow as tf

from params import Params


class TFOrnsteinUhlenbeckActionNoise(tf.Module):
    # Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

    def __init__(self):
        super().__init__(name="TFOrnsteinUhlenbeckActionNoise")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            # std values: sigma=0.3, theta=.15, dt=1e-2, decay=6e-5, x0=None
            self.dtype = Params.DTYPE
            self.theta = Params.NOISE_THETA
            self.mu = tf.fill(Params.ENV_ACT_SPACE, value=Params.NOISE_MU)
            self.sigma = tf.Variable(Params.NOISE_SIGMA)
            self.sigma_min = 5e-3
            self.dt = Params.DT
            self.x0 = Params.NOISE_X0
            self.x_prev = tf.Variable((tf.zeros_like(self.mu) if self.x0 == 0. else self.x0))
            self.decay = Params.NOISE_DECAY

    def __call__(self):
        print("retracing noise call")
        with tf.device(self.device), self.name_scope:
            tf.cond(tf.math.less(self.sigma, self.sigma_min), tf.no_op, lambda: self.sigma.assign(tf.math.multiply(self.sigma, self.decay), read_value=False))
            x = tf.math.add(
                tf.math.add(self.x_prev, tf.math.multiply(tf.math.multiply(self.theta, tf.math.subtract(self.mu, self.x_prev)), self.dt)),
                tf.math.multiply(tf.math.multiply(self.sigma, tf.sqrt(self.dt)), tf.random.normal(self.mu.shape))
            )
            self.x_prev.assign(x)
            return tf.cast(x, self.dtype)

    def decrease_sigma(self):
        pass


class GaussianNoise(tf.Module):

    def __init__(self):
        super().__init__(name="GaussianNoise")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            self.mu = tf.fill(Params.ENV_ACT_SPACE, value=Params.NOISE_MU)
            self.shape = Params.ENV_ACT_SPACE
            self.sigma = tf.Variable(Params.NOISE_SIGMA)
            self.sigma_min = Params.NOISE_SIGMA_MIN
            self.decay = Params.NOISE_DECAY
            self.bound = Params.ENV_ACT_BOUND

    def __call__(self):
        with tf.device(self.device), self.name_scope:
            noise = tf.random.normal(self.shape) * self.sigma * self.bound + self.mu
            return noise

    def decrease_sigma(self):
        with tf.device(self.device), self.name_scope:
            tf.cond(tf.math.less(self.sigma, self.sigma_min),
                    tf.no_op, lambda: self.sigma.assign(tf.math.multiply(self.sigma, self.decay), read_value=False))

