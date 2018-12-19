from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from keras.layers.convolutional import _Conv as KerasConvBase

from keras import backend as K


class GroupedConv2D(KerasConvBase):
    '''Grouped 2D convolution layer.

    This layer splits the input tensor into `cardinality` groups along
    the channel axis and creates a convolution kernel for each group
    that is convolved with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added
    to the outputs. The outputs of each convolution are then
    concatenated.  Finally, if `activation` is not `None`, it is
    applied to the outputs as well.

    Arguments
    ---------
    filters : int
        Number of filters in total. Must be divisible by `cardinality`.
    kernel_size : int or tuple of int
        Kernel size of each convolution.
    cardinality : int
        Number of groups. Must divide both `filters` and input_channels.

    **kwargs : see keras.layers.conv.

    Notes
    -----
    Compatible only with tensorflow backend.

    References
    ----------

    [1] Xie et al. Aggregated residual transformations for DNNs, CVPR 2017

    '''

    def __init__(self,
                 filters,
                 kernel_size,
                 cardinality,
                 strides=1,
                 padding='same',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupedConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.cardinality = cardinality

    def build(self, input_shape, **kwargs):
        '''construct layer and initialize all variables.

        '''
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        elif self.data_format == 'channels_last':
            self.channel_axis = -1
        else:
            raise NotImplementedError('data_format={} is unknown.')

        if input_shape[self.channel_axis] % self.cardinality != 0:
            raise ValueError(
                'input_shape[channels] must be divisible by cardinality. '
                '{} % {} != 0'.format(input_shape[self.channel_axis],
                                      self.cardinality))
        if self.filters % self.cardinality != 0:
            raise ValueError('Filters must be divisible by cardinality. '
                             '{} % {} != 0'.format(self.filters,
                                                   self.cardinality))

        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis] // self.cardinality

        kernel_shape = self.kernel_size + (input_dim,
                                           self.filters // self.cardinality)

        # initialize kernels and biases for each group
        self.kernels = [
            self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                name='kernel',
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)
            for _ in range(self.cardinality)
        ]

        if self.use_bias:
            self.biases = [
                self.add_weight(
                    shape=(self.filters // self.cardinality, ),
                    initializer=self.bias_initializer,
                    name='bias',
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
                for _ in range(self.cardinality)
            ]
        else:
            self.biases = None

        self.built = True

    def call(self, inputs):
        '''
        '''
        # split into groups and apply convolution to each one.
        groups = []
        for idx, split in enumerate(
                tf.split(
                    value=inputs,
                    num_or_size_splits=self.cardinality,
                    axis=self.channel_axis)):

            output = K.conv2d(
                split,
                self.kernels[idx],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

            if self.use_bias:
                output = K.bias_add(
                    output, self.biases[idx], data_format=self.data_format)
            groups.append(output)

        # concatenate again and apply activation.
        outputs = K.concatenate(groups, axis=self.channel_axis)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        '''
        '''
        config = super(GroupedConv2D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        config['cardinality'] = self.cardinality
        return config
