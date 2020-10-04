import tensorflow as tf


class Params:

    DEVICE = "GPU:0"
    DTYPE = 'float32'
    SEED = 123
    NUM_ACTORS = 4  # number of parallel distributed actors
    UPDATE_ACTOR_FREQ = 5  # actor parameter update every n episodes

    ## Environment params
    ENV_NAME = "GYM"  # GYM or SC
    ENV_OBS_SPACE = tf.TensorShape((3,))
    ENV_ACT_SPACE = tf.TensorShape((1,))
    ENV_ACT_BOUND = tf.constant([2.])
    # Lower and upper bounds of critic value output distribution (varies with environment)
    # V_min and V_max should be chosen based on the range of normalised reward values in the chosen env
    # A good rule of thumb is to set vmax to the discounted sum of the maximum instantaneous rewards for the maximum episode length; then set vmin to -vmax.
    ENV_V_MIN = tf.constant(-400.)
    ENV_V_MAX = tf.constant(+0.)
    ENV_REWARD_INF = tf.constant(999.)

    MAX_STEPS_TRAIN = tf.constant(100000)  # total number of steps to train for
    MAX_EP_STEPS = tf.constant(1000)  # max steps per episode
    WARM_UP_STEPS = tf.constant(500)  # number of steps to perform a randomly chosen action for each actor before predicting by actor

    ## Replay Buffer
    BUFFER_TYPE = "ReverbPrioritized"  # Uniform, ReverbUniform, ReverbPrioritized
    BUFFER_SIZE = tf.constant(1000000, dtype=tf.int32)  # must be power of 2 for PER
    BUFFER_PRIORITY_ALPHA = tf.constant(0.6)  # (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    BUFFER_PRIORITY_BETA_START = tf.constant(0.4, dtype=tf.float64)  # (0 - no bias correction, 1 - full bias correction)
    BUFFER_PRIORITY_BETA_END = tf.constant(1.0, dtype=tf.float64)
    BUFFER_PRIORITY_EPSILON = tf.constant(0.00001)

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
    RECORD_VIDEO = tf.constant(False)
    RECORD_VIDEO_TYPE = "GIF"  # GIF or MP4
    FRAME_SIZE = tf.constant([64., 64.])
    RECORD_FREQ = tf.constant(50)  # record episodes and save to video file every n epsidoes
    RECORD_START_EP = tf.constant(300)  # start recording at episode n
    RECORD_STEP_FREQ = tf.constant(3)  # do record step every n steps (to skip steps in between)

    LOG_TENSORBOARD = tf.constant(True)  # start with: $ tensorboard --logdir logs --reload_interval 5
    LOG_CONSOLE = tf.constant(False)  # print logs to console
    ACTOR_LOG_STEPS = tf.constant(50)  # log actor status every n episodes steps
    LEARNER_LOG_STEPS = tf.constant(200)  # log learner status every n learner steps
    TENSORFLOW_PROFILER = tf.constant(False)
    PLOT_MODELS = tf.constant(False)  # plot model summary

    # Calculate some params
    BUFFER_PRIORITY_BETA_INCREMENT = tf.divide((BUFFER_PRIORITY_BETA_END - BUFFER_PRIORITY_BETA_START),
                                               tf.cast(MAX_STEPS_TRAIN, BUFFER_PRIORITY_BETA_START.dtype))
    BUFFER_FROM_REVERB = True if BUFFER_TYPE in ("ReverbUniform", "ReverbPrioritized") else False
    BUFFER_IS_PRIORITIZED = True if BUFFER_TYPE == "ReverbPrioritized" else False
    if BUFFER_FROM_REVERB:
        BUFFER_DATA_SPEC = (
            tf.TensorSpec(ENV_OBS_SPACE, dtype=DTYPE, name="state"),
            tf.TensorSpec(ENV_ACT_SPACE, dtype=DTYPE, name="action"),
            tf.TensorSpec((N_STEP_RETURNS,), dtype=DTYPE, name="rewards_stack"),
            tf.TensorSpec((), dtype=tf.bool, name="terminal"),
            tf.TensorSpec(ENV_OBS_SPACE, dtype=DTYPE, name="state2"),
            tf.TensorSpec((NUM_ATOMS,), dtype=DTYPE, name="target_z_atoms"),
        )
        if BUFFER_IS_PRIORITIZED:
            BUFFER_PRIORITY_TABLE_NAMES = tf.constant([BUFFER_TYPE, BUFFER_TYPE + "_max", BUFFER_TYPE + "_min"])
    else:
        BUFFER_DATA_SPEC = (
            tf.TensorSpec(ENV_OBS_SPACE, dtype=DTYPE, name="state"),
            tf.TensorSpec(ENV_ACT_SPACE, dtype=DTYPE, name="action"),
            tf.TensorSpec((1,), dtype=DTYPE, name="reward"),
            tf.TensorSpec((1,), dtype=tf.bool, name="terminal"),
            tf.TensorSpec(ENV_OBS_SPACE, dtype=DTYPE, name="state2"),
            tf.TensorSpec((1,), dtype=DTYPE, name="gamma**N"),
        )
    BUFFER_DATA_SPEC_DTYPES = tuple(spec.dtype for spec in BUFFER_DATA_SPEC)
    BUFFER_DATA_SPEC_SHAPES = tuple(spec.shape for spec in BUFFER_DATA_SPEC)
    GAMMAS = tf.vectorized_map(
        lambda n, gamma=GAMMA: tf.math.pow(gamma, n),
        tf.range(N_STEP_RETURNS, dtype=DTYPE)
    )
    GAMMAS2 = tf.repeat(GAMMAS, 2)
    Z_ATOMS = tf.linspace(ENV_V_MIN, ENV_V_MAX, NUM_ATOMS)
    Z_ATOMS_ZEROS = tf.zeros_like(Z_ATOMS)
    DO_LOGGING = tf.logical_or(LOG_TENSORBOARD, LOG_CONSOLE)



