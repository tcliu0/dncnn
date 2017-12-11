import os
import json
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from models import get_model
from dataset import load_dataset

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string("model", "", "Model name (required)")
tf.app.flags.DEFINE_integer('height', 512, 'Image height in pixels.')
tf.app.flags.DEFINE_integer('width', 512, 'Image width in pixels.')
tf.app.flags.DEFINE_integer('channels', 3, 'Number of channels in image.')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_bool("test_set", False, "Use test set.")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def main(_):
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(os.path.join(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    train, dev, test, loader = load_dataset()

    denoise = get_model(FLAGS.model)(FLAGS)
    
    train_dir = os.path.join(FLAGS.train_dir, denoise.model_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        initialize_model(sess, denoise, train_dir)
        
        total_loss, metrics = denoise.evaluate(sess, (test if FLAGS.test_set else dev, loader))
        avg_psnr = sum([p for _, p, s in metrics]) / len(metrics)
        avg_ssim = sum([s for _, p, s in metrics]) / len(metrics)
        print 'Total loss: %f' % total_loss
        print 'Average PSNR: %f' % avg_psnr
        print 'Average SSIM: %f' % avg_ssim
        for iso in ['iso400', 'iso1600', 'iso6400', 'iso25k6', 'iso102k']:
            psnr = [p for ((_, i, _, _), p, s) in metrics if i == iso]
            ssim = [s for ((_, i, _, _), p, s) in metrics if i == iso]
            print '%s: %f\t%f' % (iso, sum(psnr) / len(psnr), sum(ssim) / len(ssim))

if __name__ == '__main__':
    tf.app.run()    
