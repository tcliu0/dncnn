import numpy as np
import tensorflow as tf

from denoise_model import DenoiseSystem

class MedianSystem(DenoiseSystem):
    def __init__(self, flags, K=11):
        self.model_name = 'median_denoise'
        self.K = K
        super(MedianSystem, self).__init__(flags)

    def setup_system(self):
        self.v = tf.get_variable('v', (1, ), initializer=tf.constant_initializer(1.0))
        
        rs = np.zeros((self.K, self.K, 1, self.K*self.K))
        for u in range(self.K):
            for v in range(self.K):
                rs[u, v, 0, self.K*u+v] = 1
                
        f_r = rs * np.array([1, 0, 0]).reshape((3, 1))
        f_g = rs * np.array([0, 1, 0]).reshape((3, 1))
        f_b = rs * np.array([0, 0, 1]).reshape((3, 1))
        
        r = tf.nn.conv2d(self.im_placeholder, f_r, strides=[1, 1, 1, 1], padding='SAME')
        g = tf.nn.conv2d(self.im_placeholder, f_g, strides=[1, 1, 1, 1], padding='SAME')
        b = tf.nn.conv2d(self.im_placeholder, f_b, strides=[1, 1, 1, 1], padding='SAME')
        r = tf.nn.top_k(r, self.K*self.K/2+1).values[:,:,:,self.K*self.K/2:self.K*self.K/2+1]
        g = tf.nn.top_k(g, self.K*self.K/2+1).values[:,:,:,self.K*self.K/2:self.K*self.K/2+1]
        b = tf.nn.top_k(b, self.K*self.K/2+1).values[:,:,:,self.K*self.K/2:self.K*self.K/2+1]
        
        self.out = self.v * tf.concat((r, g, b), axis=3)
