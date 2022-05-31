import os

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Layer, Conv2D, MaxPool2D, LeakyReLU, Flatten, Dense, Dropout, Resizing)

class LocalSubNet(Layer):
    def __init__(self, name, block_depth=16, conv_depth=32, scale=0.25):
        super(LocalSubNet, self).__init__(name=name)
        self.block_depth = block_depth
        self.conv_depth = conv_depth
        self.scale = scale
        self.layer_config = {
            "filters": self.conv_depth,
            "kernel_size": 3,
            "strides": 1,
            "padding": "same",
            "kernel_initializer": "he_normal",
            "bias_initializer": "zeros"}

    def build(self, input_shape):
        self.net = Sequential()
        # First layer
        inputs = Conv2D(
            **self.layer_config, input_shape=input_shape, 
            activation=LeakyReLU(), name="conv_0")
        self.net.add(inputs)

        # Middle layers
        for i in range(1, self.block_depth - 1):
            conv = Conv2D(
                **self.layer_config,
                activation=LeakyReLU(),
                name=f"conv_{i}")
            self.net.add(conv)

        # Last layer
        outputs = Conv2D(
            **self.layer_config, activation="tanh", 
            name=f"conv_{self.block_depth}")
        self.net.add(outputs)     

    def call(self, x):
        return self.net(x) * self.scale

class GlobalSubNet(Layer):
    def __init__(self, name, block_depth=16, conv_depth=32, image_size=128):
        super(GlobalSubNet, self).__init__(name=name)
        self.block_depth = block_depth
        self.conv_depth = conv_depth
        self.image_size = image_size
        self.layer_config = {
            "filters": self.conv_depth,
            "kernel_size": 3,
            "strides": 1,
            "padding": "same",
            "kernel_initializer": "he_normal",
            "bias_initializer": "zeros"}

    def build(self, input_shape):

        self.net = Sequential()
        # First layer
        inputs = Conv2D(
            **self.layer_config, input_shape=input_shape, 
            activation=LeakyReLU(), name="conv_0")
        self.net.add(inputs)

        # Middle layers
        for i in range(1, self.block_depth):
            conv = Conv2D(
                **self.layer_config,
                activation=LeakyReLU(),
                name=f"conv_{i}")
            maxpool = MaxPool2D(pool_size=2, strides=2, padding="same", name=f"maxpool_{i}")
            self.net.add(conv)
            self.net.add(maxpool)

        # Last layer
        self.net.add(Flatten(name="flatten"))
        self.net.add(Dense(1024, name="dense_0"))
        self.net.add(Dense(1024, name="dense_1"))
        self.net.add(Dropout(0.5, name="dropout_1"))
        self.net.add(Dense(18, name="dense_3"))
        
    def call(self, x):
        x = Resizing(
            height=self.image_size, 
            width=self.image_size,
            interpolation='bilinear')(x)
        return self.net(x)

class CIEXYZNet(Model):
    def __init__(
            self, 
            local_depth=16, 
            local_convdepth=32, 
            global_depth=5, 
            global_convdepth=64,
            global_imagesize=128, 
            scale=0.25):
        super(CIEXYZNet, self).__init__()
        self.local_depth = local_depth
        self.local_convdepth = local_convdepth
        self.global_depth = global_depth
        self.global_convdepth = global_convdepth
        self.global_imagesize = global_imagesize
        self.scale = scale

    def forward_local(self, x, target):
        if target == "xyz":
            x = self.srgb2xyz_local_net(x)
        elif target == "srgb":
            x = self.xyz2srgb_local_net(x)
        else:
            raise Exception("Wrong target")
        return x

    def forward_global(self, x, target):
        if target == "xyz":
            x_1 = self.srgb2xyz_global_net(x)
        elif target == "srgb":
            x_1 = self.xyz2srgb_global_net(x)
        else:
            raise Exception("Wrong target")

        x_1 = tf.reshape(x_1, (-1, 6, 3))
        y = tf.identity(x)
        for i in range(x_1.shape[0]):
            x_1 = tf.squeeze(x_1[i, :, :, :])
            x_1 = tf.reshape(x_1, (3, -1))
            x_1 = tf.concat(x_1, x_1*x_1)
            temp = tf.matmul(tf.squeeze(x_1[i, :, :]), x_1)
            y[i, :, :, :] = tf.reshape(temp, x.shape[1:])           
        return y

    def forward_srgb2xyz(self, srgb):
        l_xyz = srgb - self.forward_local(srgb, target='xyz')
        return self.forward_global(l_xyz, target='xyz')

    def forward_xyz2srgb(self, xyz):
        g_srgb = self.forward_global(xyz, target='srgb')
        return g_srgb + self.forward_local(g_srgb, target='srgb')

    def forward(self, x):
        xyz = self.forward_srgb2xyz(x)
        srgb = self.forward_xyz2srgb(xyz)
        return xyz, srgb

if __name__ == "__main__":
    config = {
        'local_depth': 16,
        'local_convdepth': 32,
        'global_depth': 5,
        'global_convdepth': 64,
        'global_imagesize': 128,
        'scale': 0.25}

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = CIEXYZNet(**config)
    model

