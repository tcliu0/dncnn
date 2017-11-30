import os
import re
import logging
from scipy import misc

def load_dataset():
    def loader(key):
        fname = 'dataset/%s/%s_%s_%s.bmp' % key
        im = misc.imread(fname) / 255.0
        return im

    isos = ['iso400', 'iso1600', 'iso6400', 'iso25k6', 'iso102k']
    objs = ['cereal', 'cereal2', 'chairs', 'clutter', 'clutter2',
            'clutter3', 'fan', 'flowers', 'globe', 'junkbox',
            'labbench', 'lens', 'mugpot', 'numpad', 'partsbin',
            'shoes', 'teapot', 'whtboard', 'wine', 'wirerack']
    
    train_data = []
    logging.info('Generating train tuples...')
    for obj in objs:
        for iso in isos:
            for y in range(10):
                for x in range(11):
                    im = (obj, iso, y, x)
                    gt = (obj, 'iso100', y, x)
                    train_data.append((im, gt))

    dev_data = []
    logging.info('Generating dev tuples...')
    for obj in objs:
        for iso in isos:
            for y in range(10):
                for x in [11, 12]:
                    im = (obj, iso, y, x)
                    gt = (obj, 'iso100', y, x)
                    dev_data.append((im, gt))

    test_data = []
    logging.info('Generating test tuples...')
    for obj in objs:
        for iso in isos:
            for y in range(10):
                for x in [13, 14]:
                    im = (obj, iso, y, x)
                    gt = (obj, 'iso100', y, x)
                    test_data.append((im, gt))

    logging.info('Generated %d training, %d dev and %d test tuples', len(train_data), len(dev_data), len(test_data))
    return (train_data, dev_data, test_data, loader)
