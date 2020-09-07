import tensorflow as tf


class Params:

    DEVICE = "GPU:0"
    DTYPE = 'float32'
    SEED = 123
    NUM_ACTORS = 4  # number of parallel distributed actors
    UPDATE_ACTOR_FREQ = 5  # actor parameter update every n episodes

    ## Environment params
    ENV_NAME = "GYM"  # GYM or SC
    ENV_OBS_SPACE = tf.TensorShape(3,)
    ENV_ACT_SPACE = tf.TensorShape(1,)
    ENV_ACT_BOUND = tf.constant([2.])
    # Lower and upper bounds of critic value output distribution (varies with environment)
    # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
    ENV_V_MIN = tf.constant(-20.)
    ENV_V_MAX = tf.constant(0.)

    MAX_STEPS_TRAIN = tf.constant(100000)  # total number of steps to train for
    MAX_EP_STEPS = tf.constant(1000)  # max steps per episode
    WARM_UP_STEPS = tf.constant(500)  # number of steps to perform a randomly chosen action for each actor before predicting by actor

    ## Replay Buffer
    BUFFER_TYPE = "Uniform"  # Uniform or Prioritized
    BUFFER_SIZE = tf.constant(1000000, dtype=tf.int32)  # must be power of 2 for PER
    BUFFER_PRIORITY_ALPHA = tf.constant(0.6)  # (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    BUFFER_PRIORITY_BETA_START = tf.constant(0.4)  # (0 - no bias correction, 1 - full bias correction)
    BUFFER_PRIORITY_BETA_END = tf.constant(1.0)
    BUFFER_PRIORITY_BETA_INCREMENT = (BUFFER_PRIORITY_BETA_END - BUFFER_PRIORITY_BETA_START) / tf.cast(MAX_STEPS_TRAIN, DTYPE)
    BUFFER_PRIORITY_EPSILON = tf.constant(0.00001)
    BUFFER_PARALLEL_ITERATIONS = 8  # need to be python int

    ## Networks
    MINIBATCH_SIZE = tf.constant(256, dtype=tf.int32)
    ACTOR_LEARNING_RATE = tf.constant(0.0001)
    CRITIC_LEARNING_RATE = tf.constant(0.001)
    GAMMA = tf.constant(0.99)  # Discount rate for future rewards
    TAU = tf.constant(0.001, dtype=DTYPE)  # Parameter for soft target network updates
    N_STEP_RETURNS = tf.constant(5)
    BASE_NET_ARCHITECTURE = [500, 400]  # shallow net seems to work best
    NUM_ATOMS = 51  # Number of atoms in output layer of distributional critic
    WITH_BATCH_NORM = tf.constant(True)
    WITH_DROPOUT = tf.constant(False)
    WITH_REGULARIZER = tf.constant(True)

    ## Actor Noise
    DT = tf.constant(0.02)
    NOISE_TYPE = "Gaussian"  # Gaussian or OrnsteinUhlenbeck
    NOISE_MU = tf.constant(0.)
    NOISE_SIGMA = tf.constant(0.3)
    NOISE_SIGMA_MIN = tf.constant(5e-3)  # when to stop decreasing sigma
    NOISE_THETA = tf.constant(0.15)
    NOISE_DECAY = tf.constant(0.9999)
    NOISE_X0 = tf.constant(0.)

    ## Video Recorder
    RECORD_VIDEO = tf.constant(True)
    RECORD_VIDEO_TYPE = "GIF"  # GIF or MP4
    FRAME_SIZE = tf.constant([64., 64.])
    RECORD_FREQ = tf.constant(50)  # record episodes and save to video file every n epsidoes
    RECORD_START_EP = tf.constant(300)  # start recording at episode n
    RECORD_STEP_FREQ = tf.constant(3)  # do record step every n steps (to skip steps in between)

    LOG_TENSORBOARD = tf.constant(True)  # start with $ tensorboard --logdir logs --reload_interval 2
    PLOT_MODELS = tf.constant(False)  # plot model summary

