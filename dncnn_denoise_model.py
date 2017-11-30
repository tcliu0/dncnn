import tensorflow as tf

from denoise_model import DenoiseSystem

class DnCNNDenoiseSystem(DenoiseSystem):
    def __init__(self, flags, D=17, H=32, K=3):
        self.model_name = 'dncnn_denoise'
        self.D = D
        self.H = H
        self.K = 3
        super(DnCNNDenoiseSystem, self).__init__(flags)

    def setup_system(self):
        W_1 = tf.get_variable('W_1', (self.K, self.K, self.flags.channels, self.H), initializer=tf.contrib.layers.xavier_initializer())
        b_1 = tf.get_variable('b_1', (self.H), initializer=tf.constant_initializer(0.01))
        
        h = tf.nn.relu(tf.nn.conv2d(self.im_placeholder, W_1, strides=[1, 1, 1, 1], padding='SAME') + b_1)

        for i in xrange(2, self.D):
            W = tf.get_variable('W_%d' % i, (self.K, self.K, self.H, self.H), initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_%d' % i, (self.H), initializer=tf.constant_initializer(0.01))
            gamma = tf.get_variable('gamma_%d' % i, (1), initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta_%d' % i, (1), initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding='SAME') + b
            mean, variance = tf.nn.moments(conv, axes=[0])
            norm = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, 1e-5)
            h = tf.nn.relu(norm)

        W_D = tf.get_variable('W_D', (self.K, self.K, self.H, self.flags.channels), initializer=tf.contrib.layers.xavier_initializer())
        b_D = tf.get_variable('b_D', (self.flags.channels), initializer=tf.constant_initializer(0.01))
        self.out = tf.nn.conv2d(h, W_D, strides=[1, 1, 1, 1], padding='SAME') + b_D

    def setup_loss(self):
        with tf.variable_scope('loss'):
            # Squared loss
            norm = tf.reduce_sum(tf.pow(self.out - (self.im_placeholder - self.gt_placeholder), 2.0), axis=3)
            self.loss = tf.reduce_mean(norm)

            # PSNR
            mse = tf.reduce_sum(norm, axis=(1, 2)) / (self.flags.height * self.flags.width)
            self.psnr = -10 * tf.log(mse) / tf.log(10.)
