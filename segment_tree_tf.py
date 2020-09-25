import tensorflow as tf


class SumTree:

    def __init__(self, capacity, parallel_iterations, p_max):
        # capacity must be power of 2
        self.parallel_iterations = parallel_iterations
        self.capacity = capacity
        self.n_nodes = tf.subtract(tf.multiply(2, capacity), 1)
        self.n_levels = tf.add(tf.cast(tf.math.log(tf.cast(capacity, tf.float32)) / tf.math.log(2.), tf.int32), 1)
        self.p_min = tf.Variable(p_max)
        self.tree = tf.Variable(
            initial_value=tf.zeros((self.n_nodes,), dtype=tf.float32),
            trainable=False,
            validate_shape=False,
            dtype=tf.float32,
            shape=(self.n_nodes,)
        )

    def write(self, indices, priorities):
        print("retracing SumTree write")
        priorities_min = tf.reduce_min(priorities)
        tf.cond(tf.less(priorities_min, self.p_min), lambda: self.p_min.assign(priorities_min, read_value=False), tf.no_op)
        tree_indices = tf.subtract(tf.add(indices, self.capacity), 1)
        self.update(tree_indices, priorities)

    def sum(self):
        print("retracing SumTree sum")
        return self.tree[0]

    def update(self, tree_indices, priorities):
        print("retracing SumTree update")
        changes = tf.subtract(priorities, self.tree.sparse_read(tree_indices))

        @tf.function
        def loop_tree_index(tree_idx):
            return tf.scan(lambda a, _: (a[1], tf.reshape(tf.math.floordiv((a[1] - 1), 2), ())),
                                       tf.zeros(self.n_levels,), (0, tf.reshape(tree_idx, ())),
                           infer_shape=False, parallel_iterations=self.parallel_iterations)[0]

        # @tf.function
        # def loop_tree_index(args):
        #     tree_idx = args[0]
        #     change = args[1]
        #     indices, size_total = tf.while_loop(
        #         lambda idx, size: tf.not_equal(idx.read(size-1), 0),
        #         lambda idx, size: (idx.write(size, tf.math.floordiv((idx.read(size-1) - 1), 2)), tf.add(size, 1)),
        #         loop_vars=[
        #             tf.TensorArray(tf.int32, size=0, dynamic_size=True).unstack(tf.expand_dims(tree_idx, axis=0)),
        #             tf.constant(1)
        #         ], parallel_iterations=self.parallel_iterations
        #     )
        #     self.tree = self.tree.scatter_add(tf.IndexedSlices(values=tf.repeat(change, repeats=size_total), indices=indices.stack()))
        #     return [tf.constant(0)]

        idices_to_update_all = tf.map_fn(
            loop_tree_index, elems=tree_indices,
            infer_shape=False, dtype=tf.int32, parallel_iterations=self.parallel_iterations
        )

        idices_to_update_all = tf.reshape(idices_to_update_all, (-1,))
        changes_all = tf.repeat(changes, repeats=self.n_levels)

        self.tree = self.tree.scatter_add(tf.IndexedSlices(values=changes_all, indices=idices_to_update_all))

    def get_leafs(self, values):
        print("retracing SumTree get_leafs")

        # indices = tf.range(64)
        # values = tf.range(64, dtype=tf.float32)
        # return indices, values

        @tf.function
        def loop_tree_search(parent_index, leaf_index, value, continue_loop):
            left_child_index = tf.add(tf.multiply(2, parent_index), 1)
            right_child_index = tf.add(left_child_index, 1)

            return tf.cond(
                tf.greater_equal(left_child_index, self.n_nodes),
                lambda: (parent_index, parent_index, value, tf.constant(False)),
                lambda: tf.cond(
                    tf.less_equal(value, self.tree.sparse_read(left_child_index)),
                    lambda: (left_child_index, leaf_index, value, continue_loop),
                    lambda: (right_child_index, leaf_index, tf.subtract(value, self.tree.sparse_read(left_child_index)), continue_loop),
                )
            )

        @tf.function
        def loop_value(value):
            _, leaf_index, leaf_value, _ = tf.while_loop(
                lambda *args: args[-1], loop_tree_search, [tf.constant(0), tf.constant(0), value, tf.constant(True)],
                parallel_iterations=self.parallel_iterations
            )

            # index = tf.add(tf.subtract(leaf_index, self.capacity), 1)
            # self.tree_search = self.tree_search.scatter_update(tf.IndexedSlices(values=leaf_value, indices=index))
            # return [index]
            # leaf_index = tf.constant(1) # tf.cast(tf.reshape(leaf_index, ()), tf.int32)
            # leaf_value = tf.constant(1.) # tf.cast(tf.reshape(leaf_value, ()), tf.float32)
            return [tf.add(tf.subtract(leaf_index, self.capacity), 1), leaf_value]

        # tf.print("asd")
        # tf.print(values.shape)

        # indices, values = \
        indices, values = tf.map_fn(
            loop_value, elems=values,
            dtype=[tf.int32, tf.float32], infer_shape=False, parallel_iterations=self.parallel_iterations,
        )
        # tf.print(value.shape)
        #
        #
        # indices = tf.where(self.tree_search)
        # # tf.print(indices)
        # values = self.tree_search.sparse_read(indices)
        return indices, values


class PriorityQueue:

    def __init__(self, capacity, parallel_iterations, p_max):
        # todo capacity is power of 2
        self.parallel_iterations = parallel_iterations
        self.capacity = capacity
        # self.p_min = tf.Variable(p_max)
        self.tree = tf.Variable(
            initial_value=tf.zeros((self.capacity,), dtype=tf.float32),
            trainable=False,
            validate_shape=False,
            dtype=tf.float32,
            shape=(self.capacity,)
        )

    def write(self, indices, priorities):
        print("retracing SumTree write")
        # priorities_min = tf.reduce_min(priorities)
        # tf.cond(tf.less(priorities_min, self.p_min), lambda: self.p_min.assign(priorities_min, read_value=False), tf.no_op)
        self.update(indices, priorities)

    def max(self):
        print("retracing SumTree sum")
        return self.tree[0]

    def update(self, tree_indices, priorities):
        print("retracing SumTree update")
        changes = tf.subtract(priorities, self.tree.sparse_read(tree_indices))

        @tf.function
        def loop_tree_index(tree_idx):
            return tf.scan(lambda a, _: (a[1], tf.reshape(tf.math.floordiv((a[1] - 1), 2), ())),
                                       tf.zeros(self.n_levels,), (0, tf.reshape(tree_idx, ())),
                           infer_shape=False, parallel_iterations=self.parallel_iterations)[0]

        # @tf.function
        # def loop_tree_index(args):
        #     tree_idx = args[0]
        #     change = args[1]
        #     indices, size_total = tf.while_loop(
        #         lambda idx, size: tf.not_equal(idx.read(size-1), 0),
        #         lambda idx, size: (idx.write(size, tf.math.floordiv((idx.read(size-1) - 1), 2)), tf.add(size, 1)),
        #         loop_vars=[
        #             tf.TensorArray(tf.int32, size=0, dynamic_size=True).unstack(tf.expand_dims(tree_idx, axis=0)),
        #             tf.constant(1)
        #         ], parallel_iterations=self.parallel_iterations
        #     )
        #     self.tree = self.tree.scatter_add(tf.IndexedSlices(values=tf.repeat(change, repeats=size_total), indices=indices.stack()))
        #     return [tf.constant(0)]

        idices_to_update_all = tf.map_fn(
            loop_tree_index, elems=tree_indices,
            infer_shape=False, dtype=tf.int32, parallel_iterations=self.parallel_iterations
        )

        idices_to_update_all = tf.reshape(idices_to_update_all, (-1,))
        changes_all = tf.repeat(changes, repeats=self.n_levels)

        self.tree = self.tree.scatter_add(tf.IndexedSlices(values=changes_all, indices=idices_to_update_all))

    def get_leafs(self, values):
        print("retracing SumTree get_leafs")

        # indices = tf.range(64)
        # values = tf.range(64, dtype=tf.float32)
        # return indices, values

        @tf.function
        def loop_tree_search(parent_index, leaf_index, value, continue_loop):
            left_child_index = tf.add(tf.multiply(2, parent_index), 1)
            right_child_index = tf.add(left_child_index, 1)

            return tf.cond(
                tf.greater_equal(left_child_index, self.n_nodes),
                lambda: (parent_index, parent_index, value, tf.constant(False)),
                lambda: tf.cond(
                    tf.less_equal(value, self.tree.sparse_read(left_child_index)),
                    lambda: (left_child_index, leaf_index, value, continue_loop),
                    lambda: (right_child_index, leaf_index, tf.subtract(value, self.tree.sparse_read(left_child_index)), continue_loop),
                )
            )

        @tf.function
        def loop_value(value):
            _, leaf_index, leaf_value, _ = tf.while_loop(
                lambda *args: args[-1], loop_tree_search, [tf.constant(0), tf.constant(0), value, tf.constant(True)],
                parallel_iterations=self.parallel_iterations
            )

            # index = tf.add(tf.subtract(leaf_index, self.capacity), 1)
            # self.tree_search = self.tree_search.scatter_update(tf.IndexedSlices(values=leaf_value, indices=index))
            # return [index]
            # leaf_index = tf.constant(1) # tf.cast(tf.reshape(leaf_index, ()), tf.int32)
            # leaf_value = tf.constant(1.) # tf.cast(tf.reshape(leaf_value, ()), tf.float32)
            return [tf.add(tf.subtract(leaf_index, self.capacity), 1), leaf_value]

        # tf.print("asd")
        # tf.print(values.shape)

        # indices, values = \
        indices, values = tf.map_fn(
            loop_value, elems=values,
            dtype=[tf.int32, tf.float32], infer_shape=False, parallel_iterations=self.parallel_iterations,
        )
        # tf.print(value.shape)
        #
        #
        # indices = tf.where(self.tree_search)
        # # tf.print(indices)
        # values = self.tree_search.sparse_read(indices)
        return indices, values






