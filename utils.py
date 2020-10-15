import tensorflow as tf

from params import Params


@tf.function(input_signature=[tf.TensorSpec((), Params.DTYPE), tf.TensorSpec((), tf.int32)])
def tf_round(x, decimals=0):
    print("retracing tf_round")
    multiplier = tf.cast(10 ** decimals, dtype=x.dtype)
    return tf.cast((x * multiplier), tf.int32) / tf.cast(multiplier, tf.int32)


def l2_project(z_p, p, z_q):
    """
    Taken from: https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py
    Projects the target distribution onto the support of the original network [Vmin, Vmax]
    ---
    Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = tf.where(d_neg > 0, 1. / d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = tf.where(d_pos > 0, 1. / d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)  # B x Kq x Kp

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p, 2)


class Counter(tf.Module):

    def __init__(self, name, start, dtype):
        super().__init__(name=name)
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            self.start = start
            self.counter = tf.Variable(start, dtype=dtype, name=name)
            self.counter_cs = tf.CriticalSection(name=name+"_cs")

    def increment(self, increment=1):
        """
        Increments counter in a thread safe manner.
        """
        with tf.device(self.device), self.name_scope:
            print("retracing Counter increment")

            def assign_add():
                return self.counter.assign_add(increment).value()

            return self.counter_cs.execute(assign_add)

    def val(self):
        """
        Get the counter value in a thread safe manner.
        """
        with tf.device(self.device), self.name_scope:
            print("retracing Counter call")

            def get_value():
                return self.counter.value()

            return self.counter_cs.execute(get_value)

    def reset(self):
        """
        Reset the counter value in a thread safe manner and returns last value.
        """
        with tf.device(self.device), self.name_scope:
            print("retracing Counter reset")

            def reset():
                val = self.counter.value()
                self.counter.assign(self.start)
                return val

            return self.counter_cs.execute(reset)

