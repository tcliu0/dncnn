import os
import logging
import tensorflow as tf

from util import Progbar, get_minibatches

class DenoiseSystem(object):
    def __init__(self, flags):
        # Save commandline parameters
        self.flags = flags

        # Set up placeholder tokens
        self.im_placeholder = tf.placeholder(tf.float32, (None, self.flags.height, self.flags.width, self.flags.channels))
        self.gt_placeholder = tf.placeholder(tf.float32, (None, self.flags.height, self.flags.width, self.flags.channels))
        self.train_phase = tf.placeholder(tf.bool)

        with tf.variable_scope('denoise'):
            self.setup_system()
            self.setup_loss()

        optimizer = tf.train.AdamOptimizer(self.flags.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        self.norm = tf.global_norm(gradients)
        self.train_op = optimizer.apply_gradients(zip(gradients, v))
        self.saver = tf.train.Saver(max_to_keep=10)

    def setup_system(self):
        raise NotImplementedError

    def setup_loss(self):
        with tf.variable_scope('loss'):
            # Squared loss
            norm = tf.reduce_sum(tf.pow(self.out - self.gt_placeholder, 2.0), axis=3)
            self.loss = tf.reduce_mean(norm)

            # PSNR
            mse = tf.reduce_sum(norm, axis=(1, 2)) / (self.flags.height * self.flags.width)
            self.psnr = -10 * tf.log(mse) / tf.log(10.)

    def optimize(self, session, dataset, epoch):
        input_feed = {self.train_phase: True}
        output_feed = [self.train_op, self.loss, self.norm]

        train, loader = dataset

        total_loss = 0.

        prog = Progbar(target=(len(train) - 1) / self.flags.batch_size + 1)
        for i, batch in enumerate(get_minibatches(train, self.flags.batch_size)):
            input_feed[self.im_placeholder] = [loader(b[0]) for b in batch]
            input_feed[self.gt_placeholder] = [loader(b[1]) for b in batch]

            _, loss, norm = session.run(output_feed, input_feed)
            prog.update(i+1, [("train loss", loss), ("norm", norm)])
            total_loss += loss

        return total_loss

    def predict(self, session, test_data):
        input_feed = {self.train_phase: False}
        output_feed = self.out

        test, loader = test_data

        input_feed[self.im_placeholder] = [loader(test)]

        result = session.run(output_feed, input_feed)[0]
        return result

    def evaluate(self, session, dataset):
        input_feed = {self.train_phase: False}
        output_feed = [self.loss, self.psnr]

        test, loader = dataset

        total_loss = 0.
        all_psnr = []

        prog = Progbar(target=(len(test) - 1) / self.flags.batch_size + 1)
        for i, batch in enumerate(get_minibatches(test, self.flags.batch_size, shuffle=False)):
            input_feed[self.im_placeholder] = [loader(b[0]) for b in batch]
            input_feed[self.gt_placeholder] = [loader(b[1]) for b in batch]

            loss, psnr = session.run(output_feed, input_feed)
            total_loss += loss * len(batch)
            all_psnr.extend(zip([b[0] for b in batch], psnr))
            prog.update(i+1, exact=[("total loss", total_loss)])

        return total_loss, all_psnr

    def train(self, session, dataset, start_epoch=0):
        train_dir = os.path.join(self.flags.train_dir, self.model_name)
        train, dev, test, loader = dataset
        for epoch in range(start_epoch, self.flags.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.optimize(session, (train, loader), epoch)

            logging.info("Evaluating PSNR on dev set...")
            total_loss, all_psnr = self.evaluate(session, (dev, loader))
            avg_psnr = sum([p[1] for p in all_psnr]) / len(all_psnr)
            logging.info("%.04f dB", avg_psnr)
            
            self.saver.save(session, '%s/%s.ckpt' % (train_dir, self.model_name), global_step=epoch)
            logging.info("Saving model in %s", train_dir)
