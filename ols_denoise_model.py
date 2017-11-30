import tensorflow as tf

from denoise_model import DenoiseSystem

class OLSDenoiseSystem(DenoiseSystem):
    def __init__(self, flags):
        super(OLSDenoiseSystem, self).__init__(flags)
        self.model_name = 'ols_denoise'

    def setup_system(self):
        self.W = tf.get_variable('W', (11, 11, self.flags.channels, self.flags.channels), initializer=tf.constant_initializer(0.0))
        self.out = tf.nn.conv2d(self.im_placeholder, self.W, strides=[1, 1, 1, 1], padding='SAME')

    def setup_loss(self):
        with tf.variable_scope('loss'):
            # Squared loss
            norm = tf.reduce_sum(tf.pow(self.out - self.gt_placeholder, 2.0), axis=3)
            self.loss = tf.reduce_mean(norm)

            # PSNR
            mse = tf.reduce_sum(norm, axis=(1, 2)) / (self.flags.height * self.flags.width)
            self.psnr = -10 * tf.log(mse) / tf.log(10.)
