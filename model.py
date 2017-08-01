import tensorflow as tf
# import keras


class NNCFModel():
    def __init__(self, args):
        self.args = args
        if args.sampling_strategy == 'negative':
            self.user_num = args.b
            self.item_num = args.b * (1 + args.k)
        elif args.sampling_strategy == 'stratified_sampling':
            self.user_num = args.b * (1 + args.k)
            self.item_num = args.b / args.s

        self.x_u_all = tf.placeholder(name='x_u', dtype=tf.float32,
                                      shape=[self.user_num, args.user_vector_dim])
        self.x_v_all = tf.placeholder(name='x_v', dtype=tf.float32,
                                      shape=[self.item_num, args.seq_max_length, args.word_vector_dim])

        self.f_u_all = self.f(self.x_u_all)
        self.g_v_all = self.g(self.x_v_all)

        r_uv_all = tf.matmul(self.f_u_all, self.g_v_all, transpose_b=True)

        if args.sampling_strategy == 'negative':
            L_positive_mean = tf.reduce_sum(tf.log_sigmoid(tf.stack([r_uv_all[i, i * (args.k + 1)]
                                         for i in range(args.b)])))
            L_negative_mean = tf.reduce_sum(tf.log_sigmoid(
                    tf.reduce_mean(-tf.stack([r_uv_all[i, i * (args.k + 1) + 1: (i + 1) * (args.k + 1)]
                                         for i in range(args.b)]), axis=1)
                ))
            self.loss = -(L_positive_mean + args.Lambda * L_negative_mean)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def f(self, x_u):
        f_w = tf.get_variable('f_w', [self.args.user_vector_dim, self.args.embedding_dim],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.075))
        f_b = tf.get_variable('f_b', [self.args.embedding_dim],
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.075))
        output = tf.nn.relu(tf.nn.xw_plus_b(x_u, f_w, f_b))
        return output
        # x_u = tf.expand_dims(x_u, dim=2)
        # conv1 = keras.layers.Conv1D(filters=8, kernel_size=5, strides=2, activation='relu')(x_u)
        # maxpolling1 = keras.layers.MaxPool1D(pool_size=2)(conv1)
        # conv2 = keras.layers.Conv1D(filters=16, kernel_size=5, strides=2, activation='relu')(maxpolling1)
        # maxpolling2 = keras.layers.MaxPool1D(pool_size=2)(conv2)
        # conv3 = keras.layers.Conv1D(filters=8, kernel_size=5, strides=2, activation='relu')(maxpolling2)
        # maxpolling3 = keras.layers.MaxPool1D(pool_size=2)(conv3)
        # conv4 = keras.layers.Conv1D(filters=1, kernel_size=5, strides=2, activation='relu')(maxpolling3)
        # res = keras.layers.Dense(self.args.embedding_dim)(tf.squeeze(conv4))
        # return res

    def g(self, x_v):
        x_v_means = tf.reduce_mean(x_v, axis=1)
        g_w = tf.get_variable('g_w', [self.args.word_vector_dim, self.args.embedding_dim],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.075))
        g_b = tf.get_variable('g_b', [self.args.embedding_dim],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.075))
        output = tf.nn.relu(tf.nn.xw_plus_b(x_v_means, g_w, g_b))
        return output
        # return tf.zeros(shape=[self.item_num, self.args.embedding_dim])
        # conv1 = keras.layers.Conv1D(filters=8, kernel_size=3, strides=1, activation='relu', padding='same')(x_v)
        # conv2 = keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same')(conv1)
        # conv3 = keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, activation='relu', padding='same')(conv2)
        # cell = tf.nn.rnn_cell.BasicRNNCell(self.args.embedding_dim)
        # state = cell.zero_state(batch_size=self.item_num, dtype=tf.float32)
        # for t in range(self.args.seq_max_length):
        #     output, state = cell(x_v[:, t, :], state)
        # return state
        # res = keras.layers.Dense(self.args.embedding_dim)(tf.squeeze(conv3))
        # return res
