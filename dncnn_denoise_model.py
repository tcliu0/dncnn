import tensorflow as tf
from skimage.measure import compare_ssim as ssim

from util import Progbar, get_minibatches

from denoise_model import DenoiseSystem

class DnCNNDenoiseSystem(DenoiseSystem):
    def __init__(self, flags, D=8, H=128, K=5):
        self.model_name = 'dncnn_denoise'
        self.D = D
        self.H = H
        self.K = K
        super(DnCNNDenoiseSystem, self).__init__(flags)

    def setup_system(self):
        h = tf.contrib.layers.conv2d(self.im_placeholder, self.H, self.K, scope='layer0')
        
        for i in xrange(2, self.D):
            # if i == 2 or i == (self.D - 1):
            #     conv = tf.contrib.layers.conv2d(h,
            #                                     self.H,
            #                                     self.K,
            #                                     activation_fn=None,
            #                                     biases_initializer=None,
            #                                     scope='conv_layer%d' % i)
            #     bn_train = tf.contrib.layers.batch_norm(conv, scale=True,
            #                                             activation_fn=tf.nn.relu,
            #                                             updates_collections=None,
            #                                             is_training=True,
            #                                             reuse=False,
            #                                             scope='bn_train%d' % i)
            #     bn_inf = tf.contrib.layers.batch_norm(conv, scale=True,
            #                                           activation_fn=tf.nn.relu,
            #                                           updates_collections=None,
            #                                           is_training=False,
            #                                           reuse=True,
            #                                           scope='bn_train%d' % i)
            #     h = tf.cond(self.train_phase, lambda: bn_train, lambda: bn_inf)
            # else:
            h = tf.contrib.layers.conv2d(h,
                                         self.H,
                                         self.K,
                                         scope='conv_layer%d' % i)

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
        resid = super(DnCNNDenoiseSystem, self).predict(session, test_data)
        image, loader = test_data
        return loader(image) - resid
        #return 0.5 - resid

    def evaluate(self, session, dataset):
        input_feed = {self.train_phase: False}
        output_feed = [self.loss, self.psnr, self.out]

        test, loader = dataset

        total_loss = 0.
        metrics = []

        prog = Progbar(target=(len(test) - 1) / self.flags.batch_size + 1)
        for i, batch in enumerate(get_minibatches(test, self.flags.batch_size, shuffle=False)):
            input_feed[self.im_placeholder] = [loader(b[0]) for b in batch]
            input_feed[self.gt_placeholder] = [loader(b[1]) for b in batch]

            loss, psnr, out = session.run(output_feed, input_feed)
            total_loss += loss * len(batch)
            all_ssim = [ssim(im - resid, gt, multichannel=True) for resid, im, gt in zip(out, input_feed[self.im_placeholder], input_feed[self.gt_placeholder])]
            metrics.extend(zip([b[0] for b in batch], psnr, all_ssim))
            prog.update(i+1, exact=[("total loss", total_loss)])

        return total_loss, metrics
