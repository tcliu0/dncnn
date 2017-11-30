import tensorflow as tf

from denoise_model import DenoiseSystem

"""
Identity model, outputs the input. Useful for evaluating PSNR on noisy images
"""
class IdentitySystem(DenoiseSystem):
    def __init__(self, flags):
        super(IdentitySystem, self).__init__(flags)
        self.model_name = 'identity'

    def setup_system(self):
        self.v = tf.get_variable('v', (1, ), initializer=tf.constant_initializer(1.0))
        self.out = self.v * self.im_placeholder
