import tensorflow as tf


class ParamsTest:

    MODEL_PATH = "models/2020_10_15_01_03_00_ReverbPrioritized_B1024_5N_Gaussian_Net640-624-592"
    N_EPISODES = tf.constant(10)


class Params:

    ENABLE_XLA = False  # optimizing compiler, see https://www.tensorflow.org/xla
    # set XLA envvars: export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda;
    # to enable auto-clustering on CPU: export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

    DEVICE = "GPU:0"
    DTYPE = 'float32'  # should be float32 at least
    USE_MIXED_PRECISION = False  # set if GPU computeCapability >= 7 (https://www.tensorflow.org/guide/mixed_precision)
    SEED = 123
    NUM_ACTORS = 4  # number of parallel distributed actors
    UPDATE_ACTOR_FREQ = 5  # actor parameter update every n episodes

    MIN_STEPS_TRAIN = tf.constant(8000)  # minimum number of steps to train for (so logs will be deleted on interrupt)
    MAX_STEPS_TRAIN = tf.constant(400000)  # total number of steps to train for
    MAX_EP_STEPS = tf.constant(1200)  # max steps per episode
    WARM_UP_STEPS = tf.constant(2500)  # number of steps per actor to perform randomly chosen action before predicting

    # Environment params
    # V_min and V_max = Lower and upper bounds of critic value output distribution
    # (should be chosen based on the range of normalised reward values in the chosen env)
    # rule of thumb: V_max = discounted sum of the maximum instantaneous rewards for the maximum episode length
    # V_min = - V_max

    ENV_NAME = "GameTF"  # GameTF or GymTF
    ENV_IMAGE_INPUT = True

    if ENV_NAME == "GymTF":

        if ENV_IMAGE_INPUT:
            raise NotImplementedError
        else:
            ENV_OBS_SPACE = tf.TensorShape((3,))

        ENV_ACT_SPACE = tf.TensorShape((1,))
        ENV_ACT_BOUND = tf.constant([2.])

        ENV_V_MAX = tf.constant(+0.)
        ENV_V_MIN = tf.constant(-400.)
        ENV_REWARD_INF = tf.constant(999.)

    elif ENV_NAME == "GameTF":

        ENV_N_GOALS = tf.constant(4)

        if ENV_IMAGE_INPUT:
            ENV_OBS_SPACE = tf.TensorShape((65, 65, 3))
        else:
            ENV_OBS_SPACE = tf.TensorShape(4 + 2*ENV_N_GOALS,)

        ENV_ACT_SPACE = tf.TensorShape(2,)
        ENV_ACT_BOUND = tf.constant([1.])

        ENV_V_MAX = tf.cast(tf.constant(1 * ENV_N_GOALS), DTYPE)
        ENV_V_MIN = tf.constant(-ENV_V_MAX)
        ENV_REWARD_INF = tf.constant(999.)

    else:
        raise Exception(f"Environment with name {ENV_NAME} not found.")

    # Replay Buffer
    BUFFER_TYPE = "ReverbUniform"  # Uniform, ReverbUniform, ReverbPrioritized todo try change
    BUFFER_SIZE = tf.constant(1000000, dtype=tf.int32)  # must be power of 2 for PER
    BUFFER_PRIORITY_ALPHA = tf.constant(0.6)  # (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    BUFFER_PRIORITY_BETA_START = tf.constant(0.4, dtype=tf.float64)  # (0: no bias correction, 1: full bias correction)
    BUFFER_PRIORITY_BETA_END = tf.constant(1.0, dtype=tf.float64)
    BUFFER_PRIORITY_EPSILON = tf.constant(0.00001)

    # Networks
    MINIBATCH_SIZE = tf.constant(256, dtype=tf.int32)
    ACTOR_LEARNING_RATE = tf.constant(0.0001)
    CRITIC_LEARNING_RATE = tf.constant(0.001)
    GAMMA = tf.constant(0.996)  # Discount rate for future rewards
    TAU = tf.constant(0.001, dtype=DTYPE)  # Parameter for soft target network updates
    N_STEP_RETURNS = tf.constant(5)
    BASE_NET_ARCHITECTURE = [512]  # shallow net seems to work best, should be divisible by 16
    NUM_ATOMS = 51  # Number of atoms in output layer of distributional critic
    WITH_BATCH_NORM = tf.constant(True)
    WITH_DROPOUT = tf.constant(False)
    WITH_REGULARIZER = tf.constant(True)

    # Actor Noise
    DT = tf.constant(0.02)
    NOISE_TYPE = "Gaussian"  # Gaussian or OrnsteinUhlenbeck
    NOISE_MU = tf.constant(0.)
    NOISE_SIGMA = tf.constant(0.5)
    NOISE_SIGMA_MIN = tf.constant(5e-3)  # when to stop decreasing sigma
    NOISE_THETA = tf.constant(0.15)
    NOISE_DECAY = tf.constant(0.999253712)
    NOISE_X0 = tf.constant(0.)

    # Video Recorder
    RECORD_VIDEO = tf.constant(True)
    RECORD_VIDEO_TYPE = "GIF"  # GIF or MP4
    FRAME_SIZE = tf.constant([65., 65.])
    RECORD_START_EP = tf.constant(500)  # start recording at episode n
    RECORD_FREQ = tf.constant(150)  # record episodes and save to video file every n epsidoes
    RECORD_STEP_FREQ = tf.constant(3)  # do record step every n steps (to skip steps in between)

    LOG_TENSORBOARD = tf.constant(False)  # start with: $ tensorboard --logdir logs --reload_interval 5
    LOG_CONSOLE = tf.constant(True)  # print logs to console
    ACTOR_LOG_STEPS = tf.constant(25)  # log actor status every n episodes
    LEARNER_LOG_STEPS = tf.constant(200)  # log learner status every n learner steps
    TENSORFLOW_PROFILER = tf.constant(False)
    PLOT_MODELS = tf.constant(False)  # plot model summary
    SAVE_MODEL = tf.constant(True)  # save actor network after training

    """
    
    
    
    
    
    
    
    
    
    
    """

    # Calculate some params
    assert tf.reduce_all(tf.equal(tf.constant(ENV_OBS_SPACE[0:2], dtype=DTYPE), FRAME_SIZE[0:2])), \
        f"ENV_OBS_SPACE ({ENV_OBS_SPACE}) must match FRAME_SIZE ({FRAME_SIZE}) in first two dims"
    BUFFER_PRIORITY_BETA_INCREMENT = tf.divide((BUFFER_PRIORITY_BETA_END - BUFFER_PRIORITY_BETA_START),
                                               tf.cast(MAX_STEPS_TRAIN, BUFFER_PRIORITY_BETA_START.dtype))
    BUFFER_FROM_REVERB = tf.constant(True) if BUFFER_TYPE in ("ReverbUniform", "ReverbPrioritized") else tf.constant(False)
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



