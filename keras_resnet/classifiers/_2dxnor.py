# -*- coding: utf-8 -*-

"""
keras_resnet.classifiers
~~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular residual two-dimensional classifiers.
"""

#import keras.backend
import keras.layers
import keras.models
import keras.regularizers

import keras_resnet.models

#
  #
    #
      #
        #
          #
            #
# Adapted from:
#   https://github.com/DingKe/nn_playground/blob/master/xnornet/mnist_cnn.py
# Literature:
#   https://arxiv.org/pdf/1512.03385.pdf
#   https://arxiv.org/pdf/1611.05431.pdf
#

#from keras import layers
#from keras import models

#DingKe:
#XNOR sketching:
#binary_ops
# -*- coding: utf-8 -*-
#from __future__ import absolute_import

import numpy as np

import keras.backend as K
from keras.layers import InputSpec, Layer, Dense, Conv2D
from keras import constraints
from keras import initializers


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    '''Binary hard sigmoid for training binarized neural network.
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    '''Binary hard sigmoid for training binarized neural network.
     The neurons' activations binarization function
     It behaves like the sign function during forward propagation
     And like:
        hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
        clear gradient when |x| > 1 during back propagation
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    return 2 * round_through(_hard_sigmoid(x)) - 1


def binarize(W, H=1):
    '''The weights' binarization function, 
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    # [-H, H] -> -H or H
    Wb = H * binary_tanh(W / H)
    return Wb


def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = binarize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb

#XNOR sketching:
#binary_layers.py
# -*- coding: utf-8 -*-
#import numpy as np

#from keras import backend as K

#from keras.layers import InputSpec, Layer, Dense, Conv2D
#from keras import constraints
#from keras import initializers

#from binary_ops import binarize


class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}


class BinaryDense(Dense):
    ''' Binarized Dense layer
    References: 
    "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, units, H=1., kernel_lr_multiplier='Glorot', bias_lr_multiplier=None, **kwargs):
        super(BinaryDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        
        super(BinaryDense, self).__init__(units, **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot H: {}'.format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))
            
        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.output_dim,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None
        self.built = True


    def call(self, inputs):
        binary_kernel = binarize(self.kernel, H=self.H)
        output = K.dot(inputs, binary_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
        
    def get_config(self):
        config = {'H': self.H,
                  'W_lr_multiplier': self.W_lr_multiplier,
                  'b_lr_multiplier': self.b_lr_multiplier}
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinaryConv2D(Conv2D):
    '''Binarized Convolution2D layer
    References: 
    "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, filters, kernel_lr_multiplier='Glorot', 
                 bias_lr_multiplier=None, H=1., **kwargs):
        super(BinaryConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
            
        base = self.kernel_size[0] * self.kernel_size[1]
        if self.H == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            #print('Glorot H: {}'.format(self.H))
            
        if self.kernel_lr_multiplier == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5/ (nb_input + nb_output)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.output_dim,),
                                     initializer=self.bias_initializers,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.built = True

    def call(self, inputs):
        binary_kernel = binarize(self.kernel, H=self.H) 
        outputs = K.conv2d(
            inputs,
            binary_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(BinaryConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases

BinaryConvolution2D = BinaryConv2D


#XNOR sketching:
#xnor_layers.py

# -*- coding: utf-8 -*-
#import numpy as np
#from keras import backend as K
#from binary_ops import xnorize

#from binary_layers import BinaryDense, BinaryConv2D


class XnorDense(BinaryDense):
    '''XNOR Dense layer
    References: 
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](http://arxiv.org/abs/1603.05279)
    '''
    def call(self, inputs, mask=None):
        inputs_a, inputs_b = xnorize(inputs, 1., axis=1, keepdims=True) # (nb_sample, 1)
        kernel_a, kernel_b = xnorize(self.kernel, self.H, axis=0, keepdims=True) # (1, units)
        output = K.dot(inputs_b, kernel_b) * kernel_a * inputs_a
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class XnorConv2D(BinaryConv2D):
    '''XNOR Conv2D layer
    References: 
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](http://arxiv.org/abs/1603.05279)
    '''
    def call(self, inputs):
        _, kernel_b = xnorize(self.kernel, self.H)
        _, inputs_b = xnorize(inputs)
        outputs = K.conv2d(inputs_b, kernel_b, strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        # calculate Wa and xa
        
        # kernel_a
        mask = K.reshape(self.kernel, (-1, self.filters)) # self.nb_row * self.nb_col * channels, filters 
        kernel_a = K.stop_gradient(K.mean(K.abs(mask), axis=0)) # filters
        
        # inputs_a
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        mask = K.mean(K.abs(inputs), axis=channel_axis, keepdims=True) 
        ones = K.ones(self.kernel_size + (1, 1))
        inputs_a = K.conv2d(mask, ones, strides=self.strides,
                      padding=self.padding,
                      data_format=self.data_format,
                      dilation_rate=self.dilation_rate) # nb_sample, 1, new_nb_row, new_nb_col
        if self.data_format == 'channels_first':
            outputs = outputs * K.stop_gradient(inputs_a) * K.expand_dims(K.expand_dims(K.expand_dims(kernel_a, 0), -1), -1)
        else:
            outputs = outputs * K.stop_gradient(inputs_a) * K.expand_dims(K.expand_dims(K.expand_dims(kernel_a, 0), 0), 0)
                                
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# Aliases

XnorConvolution2D = XnorConv2D

#
  #
#

# Consider builing resnet50 with this repo:
#   https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

          #
        #
      #
    #
  #
#


class ResNet18(keras.models.Model):
    """
    A :class:`ResNet18 <ResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet18(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet18(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
#        outputs = XnorDense(classes, activation="softmax")(outputs)

        super(ResNet18, self).__init__(inputs, outputs)


class ResNet34(keras.models.Model):
    """
    A :class:`ResNet34 <ResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet34(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet34(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
#        outputs = XnorDense(classes, activation="softmax")(outputs)
    
        super(ResNet34, self).__init__(inputs, outputs)


class ResNet50(keras.models.Model):
    """
    A :class:`ResNet50 <ResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet50(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet50(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
#        outputs = XnorDense(classes, activation="softmax")(outputs)
    
        super(ResNet50, self).__init__(inputs, outputs)


class ResNet101(keras.models.Model):
    """
    A :class:`ResNet101 <ResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet101(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet101(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
#        outputs = XnorDense(classes, activation="softmax")(outputs)

        super(ResNet101, self).__init__(inputs, outputs)


class ResNet152(keras.models.Model):
    """
    A :class:`ResNet152 <ResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet152(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet152(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
#        outputs = XnorDense(classes, activation="softmax")(outputs)

        super(ResNet152, self).__init__(inputs, outputs)


class ResNet200(keras.models.Model):
    """
    A :class:`ResNet200 <ResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet200(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet200(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
#        outputs = XnorDense(classes, activation="softmax")(outputs)

        super(ResNet200, self).__init__(inputs, outputs)
