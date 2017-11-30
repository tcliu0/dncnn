import tensorflow as tf

from denoise_model import DenoiseSystem

class DnCNNDenoiseSystem(DenoiseSystem):
    def __init__(self, flags, D=17, H=16, K=3):
        self.model_name = 'dncnn_denoise'
        self.D = D
        self.H = H
        self.K = 3
        super(DnCNNDenoiseSystem, self).__init__(flags)

    def setup_system(self):
        h = tf.contrib.layers.conv2d(self.im_placeholder, self.H, self.K, scope='layer0')
        
        for i in xrange(2, self.D):
            conv = tf.contrib.layers.conv2d(h,
                                            self.H,
                                            self.K,
                                            activation_fn=None,
                                            scope='conv_layer%d' % i)
            bn_train = tf.contrib.layers.batch_norm(conv, scale=True,
                                                    updates_collections=None,
                                                    is_training=True,
                                                    scope='bn_train%d' % i)
            bn_inf = tf.contrib.layers.batch_norm(conv, scale=True,
                                                  updates_collections=None,
                                                  is_training=False,
                                                  scope='bn_inf%d' % i)
            h = tf.cond(self.train_phase, lambda: bn_train, lambda: bn_inf)

        self.out = tf.contrib.layers.conv2d(h,
                                            self.flags.channels,
                                            self.K,
                                            activation_fn=None,
                                            scope='layerD')

    def setup_loss(self):
        with tf.variable_scope('loss'):
            # Squared loss
            norm = tf.reduce_sum(tf.pow(self.out - (self.im_placeholder - self.gt_placeholder), 2.0), axis=3)
            self.loss = tf.reduce_mean(norm)

            # PSNR
            mse = tf.reduce_sum(norm, axis=(1, 2)) / (self.flags.height * self.flags.width)
            self.psnr = -10 * tf.log(mse) / tf.log(10.)

    def predict(self, session, test_data):
        result = super(DnCNNDenoiseSystem, self).predict(session, test_data)
        image, loader = test_data
        return result + loader(image)
