import os
import json
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from tqdm import tqdm

from models import get_model
from dataset import load_dataset
from util import Progbar

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
tf.app.flags.DEFINE_string("output_dir", "output", "Path to store denoised images (default: ./output)")
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

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

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

        dataset = test if FLAGS.test_set else dev
        prog = Progbar(target=len(dataset))
        for i, (image, _) in enumerate(dataset):
            result = denoise.predict(sess, (image, loader))
            filename = "%s_%s_%s_%s.bmp" % image
            misc.imsave(os.path.join(FLAGS.output_dir, filename), result[0])
            prog.update(i+1)

    print('Done!')

if __name__ == '__main__':
    tf.app.run()    
