import tensorflow as tf
import reverb

from segment_tree_tf import SumTree, PriorityQueue
from params import Params


class ReplayBuffer:

    @classmethod
    def get_replay_buffer(cls):
        print("retracing ReplayBuffer get_replay_buffer")
        if Params.BUFFER_TYPE == "Uniform":
            return UniformReplayBuffer
        elif Params.BUFFER_TYPE == "Prioritized":
            return PrioritizedReplayBufferProportional
        elif Params.BUFFER_TYPE == "ReverbUniform":
            return ReverbUniformReplayBuffer
        elif Params.BUFFER_TYPE == "ReverbPrioritized":
            return ReverbPrioritizedReplayBuffer
        else:
            raise Exception(f"Buffer with name {Params.BUFFER_TYPE} not found.")


class ReverbUniformReplayBuffer(tf.Module):

    if Params.BUFFER_TYPE == "ReverbUniform":

        device = Params.DEVICE
        name_scope = tf.name_scope("ReverbUniformReplayBuffer")

        with tf.device(device), name_scope:

            buffer_size = tf.cast(Params.BUFFER_SIZE, tf.int64)
            batch_size = tf.cast(Params.MINIBATCH_SIZE, tf.int64)
            dtypes = tuple(spec.dtype for spec in Params.BUFFER_DATA_SPEC)
            shapes = tuple(spec.shape for spec in Params.BUFFER_DATA_SPEC)

            # Initialize the reverb server
            reverb_server = reverb.Server(
                tables=[
                    reverb.Table(
                        name=Params.BUFFER_TYPE,
                        sampler=reverb.selectors.Uniform(),
                        remover=reverb.selectors.Fifo(),
                        max_size=buffer_size,
                        rate_limiter=reverb.rate_limiters.MinSize(batch_size),
                    )
                ],
            )

            # sample_client = reverb.TFClient(f"localhost:{reverb_server.port}")

            dataset = reverb.ReplayDataset(
                server_address=f'localhost:{reverb_server.port}',
                table=Params.BUFFER_TYPE,
                max_in_flight_samples_per_worker=1,  # todo params
                dtypes=dtypes,
                shapes=shapes,
            )

            dataset = dataset.batch(batch_size)
            iterator = dataset.__iter__()

    @classmethod
    def get_client(cls):
        print("retracing ReverbUniformReplayBuffer get_client")
        return reverb.TFClient(f'localhost:{cls.reverb_server.port}')

    @classmethod
    def size(cls):
        print("retracing ReverbUniformReplayBuffer size")
        return Params.MINIBATCH_SIZE

    @classmethod
    def sample_batch(cls, *args, **kwargs):
        print("retracing ReverbUniformReplayBuffer sample_batch")
        # return cls.sample_client.sample(Params.BUFFER_TYPE, cls.dtypes)[1], tf.fill((cls.batch_size,), 1.), tf.zeros((cls.batch_size,))
        return cls.iterator.get_next()[1], tf.fill((cls.batch_size,), 1.), tf.zeros((cls.batch_size,))

    @classmethod
    def update_priorities(cls, *args, **kwargs):
        print("retracing ReverbUniformReplayBuffer update_priorities")
        pass


class ReverbPrioritizedReplayBuffer(tf.Module):

    if Params.BUFFER_TYPE == "ReverbPrioritized":

        device = Params.DEVICE
        name_scope = tf.name_scope("ReverbPrioritizedReplayBuffer")

        with tf.device(device), name_scope:

            buffer_size = tf.cast(Params.BUFFER_SIZE, tf.int64)
            batch_size = tf.cast(Params.MINIBATCH_SIZE, tf.int64)
            dtypes = tuple(spec.dtype for spec in Params.BUFFER_DATA_SPEC)
            shapes = tuple(spec.shape for spec in Params.BUFFER_DATA_SPEC)

            # Initialize the reverb server
            reverb_server = reverb.Server(
                tables=[
                    reverb.Table(
                        name=Params.BUFFER_TYPE,
                        sampler=reverb.selectors.Prioritized(priority_exponent=Params.BUFFER_PRIORITY_ALPHA),
                        remover=reverb.selectors.Fifo(),
                        max_size=buffer_size,
                        rate_limiter=reverb.rate_limiters.MinSize(batch_size),
                    )
                ],
            )

            client = reverb.TFClient(f"localhost:{reverb_server.port}")

            dataset = reverb.ReplayDataset(
                server_address=f'localhost:{reverb_server.port}',
                table=Params.BUFFER_TYPE,
                max_in_flight_samples_per_worker=1,  # todo params
                dtypes=dtypes,
                shapes=shapes,
            )

            dataset = dataset.batch(batch_size)
            iterator = dataset.__iter__()

            # todo can sample in both when using own learner client

    @classmethod
    def get_client(cls):
        print("retracing ReverbPrioritizedReplayBuffer get_client")
        return reverb.TFClient(f'localhost:{cls.reverb_server.port}')

    @classmethod
    def size(cls):
        print("retracing ReverbPrioritizedReplayBuffer size")
        return Params.MINIBATCH_SIZE

    @classmethod
    def sample_batch(cls, *args, **kwargs):
        print("retracing ReverbPrioritizedReplayBuffer sample_batch")
        # info, data = cls.client.sample(Params.BUFFER_TYPE, cls.dtypes)
        info, data = cls.iterator.get_next()
        return data, tf.cast(info[1], Params.DTYPE), info[0]

    @classmethod
    def update_priorities(cls, idxes_batch, td_error_batch, *args, **kwargs):
        print("retracing ReverbPrioritizedReplayBuffer update_priorities")



class UniformReplayBuffer(tf.Module):

    if Params.BUFFER_TYPE == "Uniform":

        device = Params.DEVICE
        name_scope = tf.name_scope("UniformReplayBuffer")

        with tf.device(device), name_scope:
            data = tf.nest.map_structure(
                lambda spec: tf.Variable(
                    initial_value=tf.zeros((Params.BUFFER_SIZE, spec.shape[0]), dtype=spec.dtype),
                    trainable=False,
                    validate_shape=False,
                    dtype=spec.dtype,
                    shape=(Params.BUFFER_SIZE, spec.shape[0])
                ), Params.BUFFER_DATA_SPEC, check_types=False)
            capacity = Params.BUFFER_SIZE
            last_id = tf.Variable(-1, dtype=tf.int32, name="last_id")
            last_id_cs = tf.CriticalSection(name='last_id')

    @classmethod
    def update_priorities(cls, *args, **kwargs):
        pass

    @classmethod
    def size(cls):
        print("retracing UniformReplayBuffer size")
        return tf.minimum(cls.get_last_id() + 1, cls.capacity)

    @classmethod
    def append(cls, items):
        print("retracing UniformReplayBuffer append")
        with tf.device(cls.device), cls.name_scope:
            idx = cls.increment_last_id()
            write_row_idx = tf.expand_dims(tf.math.mod(idx, cls.capacity), axis=0)
            cls.data = [var.scatter_update(tf.IndexedSlices(tf.reshape(item, (1, -1)), write_row_idx))
                         for var, item in zip(cls.data, items)]

    @classmethod
    def sample_batch(cls, sample_batch_size, *args, **kwargs):
        print("retracing UniformReplayBuffer sample_batch")
        with tf.device(cls.device), cls.name_scope:
            with tf.name_scope('sample_batch'):
                min_val, max_val = cls.valid_range_ids(cls.get_last_id())
                ids = tf.random.uniform((sample_batch_size,), minval=min_val, maxval=max_val, dtype=tf.int32)
                rows_to_get = tf.math.mod(ids, cls.capacity)
                data = [var.sparse_read(rows_to_get) for var in cls.data]

                return data, tf.fill((sample_batch_size,), 1.), tf.zeros((sample_batch_size,))

    @classmethod
    def increment_last_id(cls, increment=1):
        with tf.device(cls.device), cls.name_scope:
            print("retracing UniformReplayBuffer increment_last_id")
            # Increments the last_id in a thread safe manner.

            def assign_add():
                return cls.last_id.assign_add(increment).value()

            return cls.last_id_cs.execute(assign_add)

    @classmethod
    def get_last_id(cls):
        with tf.device(cls.device), cls.name_scope:
            print("retracing UniformReplayBuffer get_last_id")
            # Get the last_id in a thread safe manner.

            def last_id():
                return cls.last_id.value()

            return cls.last_id_cs.execute(last_id)

    @classmethod
    def valid_range_ids(cls, last_id):
        print("retracing UniformReplayBuffer valid_range_ids")

        with tf.device(cls.device), cls.name_scope:
            min_id_not_full = tf.constant(0, dtype=tf.int32)
            max_id_not_full = tf.maximum(last_id + 1, 0)

            min_id_full = last_id + 1 - cls.capacity
            max_id_full = last_id + 1

            return tf.cond(last_id < cls.capacity, lambda: (min_id_not_full, max_id_not_full), lambda: (min_id_full, max_id_full))


class PrioritizedReplayBufferProportional(tf.Module):

    # todo # Check replay memory every REPLAY_MEM_REMOVE_STEP training steps and remove samples over REPLAY_MEM_SIZE capacity

    if Params.BUFFER_TYPE == "Prioritized":

        device = Params.DEVICE
        name_scope = tf.name_scope("PrioritizedReplayBufferProportional")

        with tf.device(device), name_scope:
            pass

#         with tf.device(self.device), self.name_scope:
#
#             self.dtype = Params.DTYPE
#             self.swap_memory = tf.constant(False)
#             self.parallel_iterations = Params.BUFFER_PARALLEL_ITERATIONS
#
#             self.alpha = Params.BUFFER_PRIORITY_ALPHA
#             self.priority_eps = Params.BUFFER_PRIORITY_EPSILON
#             self.max_priority = tf.constant(1.0)
#
#             self.capacity = Params.BUFFER_SIZE
#             self.data_table = Table(data_spec, Params.BUFFER_SIZE)
#             self.p_sum = SumTree(Params.BUFFER_SIZE, self.parallel_iterations, self.max_priority)
#
#             self.last_id = common.create_variable('last_id', -1, dtype=tf.int32)
#             self.last_id_cs = tf.CriticalSection(name='last_id')
#
#     def size(self):
#         print("retracing PrioritizedReplayBuffer size")
#         return tf.minimum(self.get_last_id() + 1, self.capacity)
#
#     def append(self, items):
#         print("retracing PrioritizedReplayBuffer append")
#         with tf.device(self.device), self.name_scope:
#             with tf.name_scope('append'):
#                 idx = self.increment_last_id()
#                 write_row_idx = tf.math.mod(idx, self.capacity)
#                 write_data_op = self.data_table.write(write_row_idx, items)
#                 priority = tf.pow(self.max_priority, self.alpha)
#                 # todo keep self.max_priority updated
#                 self.p_sum.write(tf.expand_dims(write_row_idx, axis=0), tf.expand_dims(priority, axis=0))
#                 return write_data_op
#
#     def sample_batch(self, sample_batch_size, beta):
#         print("retracing PrioritizedReplayBuffer sample_batch")
#         with tf.device(self.device), self.name_scope:
#             with tf.name_scope('sample_batch'):
#
#                 sample_batch_size = tf.cast(sample_batch_size, tf.int32)
#
#                 p_total = self.p_sum.sum()
#                 p_range = tf.divide(p_total, tf.cast(sample_batch_size, dtype=self.dtype))
#                 # todo should divide by p_total across batch_size only?
#                 p_samples = tf.random.uniform(shape=(sample_batch_size,)) * p_range + tf.range(sample_batch_size, dtype=self.dtype) * p_range
#
#                 indices, p_stack = self.p_sum.get_leafs(p_samples)
#                 # indices = tf.fill((64,), 1)
#                 rows_to_get = tf.math.mod(indices, self.capacity)
#                 data = self.data_table.read(rows_to_get)
#
#                 p_min = tf.divide(self.p_sum.p_min, p_total)
#                 max_weight = tf.pow((p_min * tf.cast(self.size(), self.dtype)), -beta)
#
#                 weights = tf.divide(tf.pow((tf.divide(p_stack, p_total) * tf.cast(self.size(), self.dtype)), -beta), max_weight)
#                 # weights = tf.fill((64,), 1.)
#
#                 return data, weights, indices
#
#     # @tf.function(input_signature=[tf.TensorSpec((64,), tf.int32), tf.TensorSpec((64,), tf.float32)])
#     def update_priorities(self, idxes, td_error):
#         print("retracing PrioritizedReplayBuffer update_priorities")
#         with tf.device(self.device), self.name_scope:
#             with tf.name_scope('update_priorities'):
#                 priorities = tf.pow((tf.abs(td_error) + self.priority_eps), self.alpha)
#                 self.p_sum.write(idxes, priorities)
#
#     def increment_last_id(self, increment=1):
#         print("retracing PrioritizedReplayBuffer increment_last_id")
#         # Increments the last_id in a thread safe manner.
#
#         def assign_add():
#             return self.last_id.assign_add(increment).value()
#
#         return self.last_id_cs.execute(assign_add)
#
#     def get_last_id(self):
#         print("retracing PrioritizedReplayBuffer get_last_id")
#         # Get the last_id in a thread safe manner.
#
#         def last_id():
#             return self.last_id.value()
#
#         return self.last_id_cs.execute(last_id)


# class PrioritizedReplayBufferRankBased(tf.Module):
#
#     # see https://github.com/evaldsurtans/dqn-prioritized-experience-replay/blob/master/rank_based.py
#     # see https://github.com/evaldsurtans/dqn-prioritized-experience-replay/blob/master/binary_heap.py
#
#     def __init__(self, data_spec):
#
#         super(PrioritizedReplayBufferRankBased, self).__init__(name="PrioritizedReplayBufferRankBased")
#         self.device = Params.DEVICE

#         with tf.device(self.device), self.name_scope:
#
#             self.dtype = Params.DTYPE
#             self.swap_memory = tf.constant(False)
#             self.parallel_iterations = Params.BUFFER_PARALLEL_ITERATIONS
#
#             self.alpha = Params.BUFFER_PRIORITY_ALPHA
#             self.priority_eps = Params.BUFFER_PRIORITY_EPSILON
#             self.max_priority = tf.constant(1.0)
#
#             self.capacity = Params.BUFFER_SIZE
#             self.data_table = Table(data_spec, Params.BUFFER_SIZE)
#             self.p_queue = PriorityQueue(Params.BUFFER_SIZE, self.parallel_iterations, self.max_priority)
#
#             self.last_id = common.create_variable('last_id', -1, dtype=tf.int32)
#             self.last_id_cs = tf.CriticalSection(name='last_id')
#
#     def size(self):
#         print("retracing PrioritizedReplayBuffer size")
#         return tf.minimum(self.get_last_id() + 1, self.capacity)
#
#     def append(self, items):
#         print("retracing PrioritizedReplayBuffer append")
#         with tf.device(self.device), self.name_scope:
#             with tf.name_scope('append'):
#                 idx = self.increment_last_id()
#                 write_row_idx = tf.math.mod(idx, self.capacity)
#                 write_data_op = self.data_table.write(write_row_idx, items)
#                 self.p_queue.write(tf.expand_dims(write_row_idx, axis=0), tf.expand_dims(self.p_queue.max(), axis=0))
#                 return write_data_op
#
#     def sample_batch(self, sample_batch_size, beta):
#         print("retracing PrioritizedReplayBuffer sample_batch")
#         with tf.device(self.device), self.name_scope:
#             with tf.name_scope('sample_batch'):
#
#                 sample_batch_size = tf.cast(sample_batch_size, tf.int32)
#
#                 idices_range = tf.divide(self.size(), tf.cast(sample_batch_size, dtype=self.dtype))
#                 idices = tf.random.uniform(shape=(sample_batch_size,)) * idices_range + tf.range(sample_batch_size, dtype=self.dtype) * idices_range
#                 idices = tf.cast(idices, tf.int32)
#
#                 indices, p_stack = self.p_sum.get_leafs(p_samples)
#                 # indices = tf.fill((64,), 1)
#                 rows_to_get = tf.math.mod(indices, self.capacity)
#                 data = self.data_table.read(rows_to_get)
#
#                 p_min = tf.divide(self.p_sum.p_min, p_total)
#                 max_weight = tf.pow((p_min * tf.cast(self.size(), self.dtype)), -beta)
#
#                 weights = tf.divide(tf.pow((tf.divide(p_stack, p_total) * tf.cast(self.size(), self.dtype)), -beta), max_weight)
#                 # weights = tf.fill((64,), 1.)
#
#                 return data, weights, indices
#
#     # @tf.function(input_signature=[tf.TensorSpec((64,), tf.int32), tf.TensorSpec((64,), tf.float32)])
#     def update_priorities(self, idxes, td_error):
#         print("retracing PrioritizedReplayBuffer update_priorities")
#         with tf.device(self.device), self.name_scope:
#             with tf.name_scope('update_priorities'):
#                 priorities = tf.pow((tf.abs(td_error) + self.priority_eps), self.alpha)
#                 self.p_sum.write(idxes, priorities)
#
#     def increment_last_id(self, increment=1):
#         print("retracing PrioritizedReplayBuffer increment_last_id")
#         # Increments the last_id in a thread safe manner.
#
#         def assign_add():
#             return self.last_id.assign_add(increment).value()
#
#         return self.last_id_cs.execute(assign_add)
#
#     def get_last_id(self):
#         print("retracing PrioritizedReplayBuffer get_last_id")
#         # Get the last_id in a thread safe manner.
#
#         def last_id():
#             return self.last_id.value()
#
#         return self.last_id_cs.execute(last_id)
