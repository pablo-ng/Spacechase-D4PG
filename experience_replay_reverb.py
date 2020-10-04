import tensorflow as tf
import reverb

from params import Params


def reduce_trajectory(trajectory):
    print("retracing reduce_trajectory")

    rewards_stack = tf.reshape(trajectory.data[2], (-1,))
    mask = tf.equal(rewards_stack, -Params.ENV_REWARD_INF)
    rewards_stack = rewards_stack[tf.logical_not(mask)]
    rewards_stack_len = tf.shape(rewards_stack)[0]

    terminal = trajectory.data[3]
    discounted_reward = tf.reduce_sum(tf.multiply(rewards_stack, Params.GAMMAS[:rewards_stack_len]))

    last_gamma = tf.math.pow(Params.GAMMA, tf.cast(rewards_stack_len, Params.DTYPE))
    _target_z_atoms = tf.cond(terminal, lambda: Params.Z_ATOMS_ZEROS, lambda: Params.Z_ATOMS)
    _target_z_atoms = discounted_reward + _target_z_atoms * last_gamma

    return trajectory.info, (
        trajectory.data[0],
        trajectory.data[1],
        tf.constant([0.]),     # arbitrary value
        tf.constant([False]),  # arbitrary value
        trajectory.data[4],
        _target_z_atoms,
    )


class ReverbUniformReplayBuffer(tf.Module):

    def __init__(self, name="ReverbUniformReplayBuffer", reverb_server=None):
        super().__init__(name=name)
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:

            self.buffer_size = tf.cast(Params.BUFFER_SIZE, tf.int64)
            self.batch_size = tf.cast(Params.MINIBATCH_SIZE, tf.int64)
            self.batch_size_float = tf.cast(Params.MINIBATCH_SIZE, tf.float64)
            self.sequence_length = tf.cast(Params.N_STEP_RETURNS, tf.int64)

            # Initialize the reverb server
            if not reverb_server:
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
            else:
                self.reverb_server = reverb_server

            dataset = reverb.ReplayDataset(
                server_address=f'localhost:{self.reverb_server.port}',
                table=Params.BUFFER_TYPE,
                max_in_flight_samples_per_worker=2*self.batch_size,
                dtypes=Params.BUFFER_DATA_SPEC_DTYPES,
                shapes=Params.BUFFER_DATA_SPEC_SHAPES,
            )

            dataset = dataset.map(
                map_func=reduce_trajectory,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=True,
            )
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(5)
            self.iterator = dataset.__iter__()

    def get_client(self):
        print("retracing ReverbUniformReplayBuffer get_client")
        return reverb.TFClient(f'localhost:{self.reverb_server.port}')

    def sample_batch(self, *args, **kwargs):
        print("retracing ReverbUniformReplayBuffer sample_batch")
        _, data = self.iterator.get_next()
        return data, tf.fill((self.batch_size,), 1.), tf.zeros((self.batch_size,))

    @staticmethod
    def update_priorities(*args, **kwargs):
        print("retracing ReverbUniformReplayBuffer update_priorities")
        pass

    @staticmethod
    def size():
        print("retracing ReverbPrioritizedReplayBuffer size")
        return Params.MINIBATCH_SIZE


class ReverbPrioritizedReplayBuffer(ReverbUniformReplayBuffer):

    def __init__(self):

        # Initialize the reverb server
        buffer_size = tf.cast(Params.BUFFER_SIZE, tf.int64)
        batch_size = tf.cast(Params.MINIBATCH_SIZE, tf.int64)
        reverb_server = reverb.Server(
            tables=[
                reverb.Table(
                    name=Params.BUFFER_TYPE,
                    sampler=reverb.selectors.Prioritized(priority_exponent=Params.BUFFER_PRIORITY_ALPHA),
                    remover=reverb.selectors.Fifo(),
                    max_size=buffer_size,
                    rate_limiter=reverb.rate_limiters.MinSize(batch_size),
                ),
                reverb.Table(
                    name=Params.BUFFER_TYPE + "_max",
                    sampler=reverb.selectors.MaxHeap(),
                    remover=reverb.selectors.Fifo(),
                    max_size=buffer_size,
                    rate_limiter=reverb.rate_limiters.MinSize(tf.constant(1)),
                ),
                reverb.Table(
                    name=Params.BUFFER_TYPE + "_min",
                    sampler=reverb.selectors.MinHeap(),
                    remover=reverb.selectors.Fifo(),
                    max_size=buffer_size,
                    rate_limiter=reverb.rate_limiters.MinSize(tf.constant(1)),
                ),
            ],
        )

        super().__init__(name="ReverbPrioritizedReplayBuffer", reverb_server=reverb_server)

        # Init client for updating priorities
        self.client = self.get_client()

        # Insert dummy trajectory
        self.client.insert(
                    [tf.zeros(spec.shape, dtype=spec.dtype) for spec in Params.BUFFER_DATA_SPEC],
                    tables=Params.BUFFER_PRIORITY_TABLE_NAMES,
                    priorities=tf.constant([1., 1., 1.], dtype=tf.float64)
                )

    def sample_batch(self, _, beta):
        print("retracing ReverbPrioritizedReplayBuffer sample_batch")
        info, data = self.iterator.get_next()
        p_min = self.client.sample(Params.BUFFER_TYPE + "_min", Params.BUFFER_DATA_SPEC_DTYPES).info.probability
        max_weight = tf.pow((tf.multiply(p_min, self.batch_size_float)), -beta)
        weights = tf.divide(tf.pow(tf.multiply(self.batch_size_float, info.probability), -beta), max_weight)
        weights = tf.cast(weights, Params.DTYPE)
        return data, weights, info.key

    def update_priorities(self, idxes_batch, td_error_batch):
        print("retracing ReverbPrioritizedReplayBuffer update_priorities")
        priorities = tf.pow((tf.abs(td_error_batch) + Params.BUFFER_PRIORITY_EPSILON), Params.BUFFER_PRIORITY_ALPHA)
        priorities = tf.cast(priorities, tf.float64)
        # tf.while_loop(
        #
        # )
        self.client.update_priorities(Params.BUFFER_PRIORITY_TABLE_NAMES[0], keys=idxes_batch, priorities=priorities)
        self.client.update_priorities(Params.BUFFER_PRIORITY_TABLE_NAMES[1], keys=idxes_batch, priorities=priorities)
        self.client.update_priorities(Params.BUFFER_PRIORITY_TABLE_NAMES[2], keys=idxes_batch, priorities=priorities)






