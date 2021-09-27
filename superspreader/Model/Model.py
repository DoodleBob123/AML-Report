""" 
The MultiConvLayer, MultiConvNet and SummaryNet Classes have been obtained
from the Code of Radev et. al (https://arxiv.org/pdf/2010.00300.pdf) from
https://github.com/stefanradev93/AIAgainstCorona. 
"""

import tensorflow as tf
import numpy as np


class MultiConvLayer(tf.keras.Model):
    """Implements an inception-inspired conv layer using different kernel sizes"""
    def __init__(self, n_filters=32, strides=1):
        super(MultiConvLayer, self).__init__()
        
        self.convs = [
            tf.keras.layers.Conv1D(n_filters//2, kernel_size=f, strides=strides, 
                                   padding='causal', activation='relu', kernel_initializer='glorot_uniform')
            for f in range(2, 8)
        ]
        self.dim_red = tf.keras.layers.Conv1D(n_filters, 1, 1, 
                                              activation='relu', kernel_initializer='glorot_uniform')
        
    def call(self, x):
        """x is a timeseries of dimensions B timestamps, n_features"""
        
        out = tf.concat([conv(x) for conv in self.convs], axis=-1)
        out = self.dim_red(out)
        return out
    
    
class MultiConvNet(tf.keras.Model):
    """Implements an inception-inspired conv layer using different kernel sizes"""
    def __init__(self, n_layers=3, n_filters=64, strides=1):
        super(MultiConvNet, self).__init__()
        
        self.net = tf.keras.Sequential([
            MultiConvLayer(n_filters, strides)
            for _ in range(n_layers)
        ])
        
        self.lstm = tf.keras.layers.LSTM(n_filters)
        
    def call(self, x, **args):
        """x is a timeseries of dimensions B timestamps, n_features"""
        
        out = self.net(x)
        out = self.lstm(out)
        return out
    
    
class SummaryNet(tf.keras.Model):
    def __init__(self, n_summary):
        super(SummaryNet, self).__init__()
        self.net_I = MultiConvNet(n_filters=n_summary//2)
        self.net_D = MultiConvNet(n_filters=n_summary//2)
    
    def call(self, x, **args):
        """x is a timeseries of dimensions B timestamps, n_features"""
        
        x = tf.split(x, 2, axis=-1)
        x_i = self.net_I(x[0])
        x_d = self.net_D(x[1])
        return tf.concat([x_i, x_d], axis=-1)


class GenerativeModel(tf.keras.Model):
    """ Generative Model that implements a summary network with invertible network."""
    
    def __init__ (self, invertible_net:callable, summary_net:callable) -> None:
        
        super(GenerativeModel, self).__init__()
        self.summary_net    = summary_net
        self.invertible_net = invertible_net
    
    def call(self, x, y, inverse:bool = False):
        """ Runs a forward or backward pass."""
        if not inverse:
            return self.forward(x, y)
        else:
            return self.inverse(x, y)
        
    def forward(self, x, y):
        """ Runs the forward pass. """
        y_summary = self.summary_net(y)
        return self.invertible_net(x, y_summary, inverse=False)
    
    def inverse(self, z, y):
        """ Runs the inverse pass. """
        y_summary = self.summary_net(y)
        return self.invertible_net(z, y_summary, inverse=True) 
    
    def sample(self, y, n_samples, to_numpy = True):
        """ Samples from the generative model. """
        y_summary = self.summary_net(y)
        return self.invertible_net.sample(y_summary, n_samples, to_numpy)

