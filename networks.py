import tensorflow as tf

from params import Params
from utils import l2_project


def plot_model(model):
    tf.keras.utils.plot_model(model, to_file='network.png', show_shapes=True, show_layer_names=True)
    print(model.summary())


@tf.function()
def update_weights(target_weights, source_weights, tau):
    print("retracing update_weights")
    assert len(target_weights) == len(source_weights)
    for tw, sw in zip(target_weights, source_weights):
        tf.keras.backend.update(tw, tau * sw + (1. - tau) * tw)


def base_net(x):
    # todo BN as first
    dropout = (lambda inp: tf.keras.layers.Dropout(rate=0.16)(inp)) if Params.WITH_DROPOUT else lambda inp: inp
    batch_norm = (lambda inp: tf.keras.layers.BatchNormalization()(inp)) if Params.WITH_BATCH_NORM else lambda inp: inp
    regularizer = tf.keras.regularizers.l2(0.02) if Params.WITH_REGULARIZER else None

    x = batch_norm(x)
    for n_units in Params.BASE_NET_ARCHITECTURE:
        x = tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=regularizer)(x)
        x = batch_norm(x)
        x = dropout(x)

    return x, regularizer


# noinspection PyMethodMayBeStatic
class ActorNetwork(tf.Module):

    def __init__(self, with_target_net, name="ActorNetwork"):
        super(ActorNetwork, self).__init__(name)
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            # Set up actor net
            self.actor_network = self.build_model()
            self.actor_network.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Params.ACTOR_LEARNING_RATE), loss='mse')
            self.tvariables = self.actor_network.trainable_variables
            self.nvariables = self.actor_network.non_trainable_variables

            if with_target_net:
                # Clone to actor net to target actor net
                self.target_actor_network = self.build_model()
                self.target_tvariables = self.target_actor_network.trainable_variables
                self.target_nvariables = self.target_actor_network.non_trainable_variables
                update_weights(self.target_tvariables + self.target_nvariables, self.tvariables + self.nvariables, tf.constant(1.))

                # Compile target; Optimizer and loss are arbitrary (needed for feed forward)
                self.target_actor_network.compile(optimizer='sgd', loss='mse')

            if Params.PLOT_MODELS:
                plot_model(self.actor_network)

    # noinspection PyTypeChecker
    def build_model(self):
        print("retracing actor build_model")
        with tf.device(self.device), self.name_scope:
            inputs = tf.keras.Input(shape=Params.ENV_OBS_SPACE)
            x, regularizer = base_net(inputs)
            x = tf.keras.layers.Dense(Params.ENV_ACT_SPACE.dims[0], activation='tanh', kernel_regularizer=regularizer,
                                      kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003))(x)
            x = tf.keras.layers.Multiply()([x, Params.ENV_ACT_BOUND])
            model = tf.keras.Model(inputs, x)
            return model

    def predict_action(self, state):
        print("retracing predict_action")
        with tf.device(self.device), self.name_scope:
            return tf.cast(self.actor_network(state, training=False), Params.DTYPE)


# noinspection PyMethodMayBeStatic
class CriticNetwork(tf.Module):

    def __init__(self):
        super(CriticNetwork, self).__init__(name="CriticNetwork")
        self.device = Params.DEVICE

        with tf.device(self.device), self.name_scope:
            # Set up critic net
            self.critic_network = self.build_model()
            self.critic_network.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Params.CRITIC_LEARNING_RATE),
                # loss=tf.keras.losses.CategoricalCrossentropy(reduction="none", from_logits=False)  # from logits as y_pred encodes a probability distribution
                loss=tf.nn.softmax_cross_entropy_with_logits
            )
            self.tvariables = self.critic_network.trainable_variables
            self.nvariables = self.critic_network.non_trainable_variables

            # Clone net to target critic net
            self.target_critic_network = self.build_model()
            self.target_tvariables = self.target_critic_network.trainable_variables
            self.target_nvariables = self.target_critic_network.non_trainable_variables
            update_weights(self.target_tvariables + self.target_nvariables, self.tvariables + self.nvariables, tf.constant(1.))

            # Compile target; Optimizer and loss are arbitrary (needed for feed forward)
            self.target_critic_network.compile(optimizer='sgd', loss='mse')

            # Init Z-Atoms once
            self.z_atoms = tf.linspace(Params.ENV_V_MIN, Params.ENV_V_MAX, Params.NUM_ATOMS)
            self.target_z_atoms = tf.linspace(Params.ENV_V_MIN, Params.ENV_V_MAX, Params.NUM_ATOMS)

            if Params.PLOT_MODELS:
                plot_model(self.critic_network)

    # noinspection PyTypeChecker
    def build_model(self):
        print("retracing critic build_model")
        with tf.device(self.device), self.name_scope:
            inputs = tf.keras.Input(shape=Params.ENV_OBS_SPACE)
            action = tf.keras.Input(shape=Params.ENV_ACT_SPACE, name="action")
            x = tf.keras.layers.Concatenate()([inputs, action])
            x, _ = base_net(x)
            # todo concat actions after base layer?
            output_logits = tf.keras.layers.Dense(Params.NUM_ATOMS, activation="linear",
                                                  kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003),
                                                  bias_initializer=tf.random_uniform_initializer(-0.003, 0.003), name="test")(x)
            output_probs = tf.keras.layers.Softmax()(output_logits)

            model = tf.keras.Model(inputs=[inputs, action], outputs=[output_probs, output_logits])
            return model

    def train(self, x, target_z_atoms, target_q_probs, is_weights):
        print("retracing critic train")
        with tf.device(self.device), self.name_scope:

            target_z_projected = l2_project(target_z_atoms, target_q_probs, self.target_z_atoms)

            with tf.GradientTape() as tape:
                y_ = self.critic_network(x, training=True)[1]
                # loss_value = self.critic_network.loss(y_true=tf.stop_gradient(target_z_projected), y_pred=y_)
                loss_value = self.critic_network.loss(labels=tf.stop_gradient(target_z_projected), logits=y_)
                weighted_loss = tf.multiply(loss_value, is_weights)
                mean_loss = tf.reduce_mean(weighted_loss)  # todo could add reduction to loss (to network loss (keras))
                l2_reg_loss = 0.  # could add L2 weight regularisation
                total_loss = mean_loss + l2_reg_loss

            grads = tape.gradient(total_loss, self.tvariables)
            self.critic_network.optimizer.apply_gradients(zip(grads, self.tvariables))
            return loss_value



