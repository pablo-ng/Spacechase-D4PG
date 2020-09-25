import tensorflow as tf
import reverb

from params import Params


class ReverbUniformReplayBuffer(tf.Module):

    def __init__(self):
        super().__init__(name="ReverbUniformReplayBuffer")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            self.buffer_size = tf.cast(Params.BUFFER_SIZE, tf.int64)
            self.batch_size = tf.cast(Params.MINIBATCH_SIZE, tf.int64)
            self.sequence_length = tf.cast(Params.N_STEP_RETURNS, tf.int64)

            # Initialize the reverb server
            self.reverb_server = reverb.Server(
                tables=[
                    reverb.Table(
                        name=Params.BUFFER_TYPE,
                        sampler=reverb.selectors.Uniform(),
                        remover=reverb.selectors.Fifo(),
                        max_size=self.buffer_size,
                        rate_limiter=reverb.rate_limiters.MinSize(self.batch_size),
                    )
                ],
            )

            dataset = reverb.ReplayDataset(
                server_address=f'localhost:{self.reverb_server.port}',
                table=Params.BUFFER_TYPE,
                max_in_flight_samples_per_worker=2*self.batch_size,  # todo params
                dtypes=tuple(spec.dtype for spec in Params.BUFFER_DATA_SPEC),
                shapes=tuple(spec.shape for spec in Params.BUFFER_DATA_SPEC),
            )

            dataset = dataset.map(self.reduce_trajectory, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
            # todo interleave
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(4)
            self.iterator = dataset.__iter__()

    def get_client(self):
        print("retracing ReverbUniformReplayBuffer get_client")
        return reverb.TFClient(f'localhost:{self.reverb_server.port}')

    def sample_batch(self, *args, **kwargs):
        print("retracing ReverbUniformReplayBuffer sample_batch")
        return self.iterator.get_next(), tf.fill((self.batch_size,), 1.), tf.zeros((self.batch_size,))

    @staticmethod
    def update_priorities(*args, **kwargs):
        print("retracing ReverbUniformReplayBuffer update_priorities")
        pass

    @staticmethod
    def size():
        print("retracing ReverbPrioritizedReplayBuffer size")
        return Params.MINIBATCH_SIZE
    
    @staticmethod
    def reduce_trajectory(trajectory):
        print("retracing ReverbPrioritizedReplayBuffer reduce_trajectory")
        last_gamma = tf.math.pow(Params.GAMMA, tf.cast(Params.N_STEP_RETURNS, Params.DTYPE))
        target_z_atoms = tf.linspace(Params.ENV_V_MIN, Params.ENV_V_MAX, Params.NUM_ATOMS)
        target_z_atoms_zeros = tf.zeros_like(target_z_atoms)

        terminal = trajectory.data[3]
        discounted_reward = tf.reduce_sum(tf.multiply(tf.squeeze(trajectory.data[2]), Params.GAMMAS))
        _target_z_atoms = tf.cond(terminal, lambda: target_z_atoms_zeros, lambda: target_z_atoms)
        _target_z_atoms = discounted_reward + _target_z_atoms * last_gamma

        return (
            trajectory.data[0],
            trajectory.data[1],
            tf.constant([0.]),
            tf.constant([False]),
            trajectory.data[4],
            _target_z_atoms,
        )


#             # Initialize the reverb server
#             reverb_server = reverb.Server(
#                 tables=[
#                     reverb.Table(
#                         name=Params.BUFFER_TYPE,
#                         sampler=reverb.selectors.Prioritized(priority_exponent=Params.BUFFER_PRIORITY_ALPHA),
#                         remover=reverb.selectors.Fifo(),
#                         max_size=buffer_size,
#                         rate_limiter=reverb.rate_limiters.MinSize(batch_size),
#                     )
#                 ],
#             )
#
#     @classmethod
#     def sample_batch(self, *args, **kwargs):
#         print("retracing ReverbPrioritizedReplayBuffer sample_batch")
#         # info, data = self.client.sample(Params.BUFFER_TYPE, self.dtypes)
#         info, data = self.iterator.get_next()
#         return data, tf.cast(info[1], Params.DTYPE), info[0]
#
#     @classmethod
#     def update_priorities(self, idxes_batch, td_error_batch, *args, **kwargs):
#         print("retracing ReverbPrioritizedReplayBuffer update_priorities")



