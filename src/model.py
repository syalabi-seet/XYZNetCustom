import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, LeakyReLU, Flatten, Dense, Dropout, Resizing, Rescaling, Concatenate)

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class CosineDecayWithWarmup(LearningRateSchedule):
    def __init__(
            self, 
            epochs,
            lr_max,
            lr_min,
            warmup_epochs=0, 
            sustain_epochs=0,            
            lr_start=1e-5,             
            n_cycles=0.5):
        self.warmup_epochs = warmup_epochs
        self.sustain_epochs = sustain_epochs
        self.epochs = epochs
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.n_cycles = n_cycles

    def __call__(self, epoch):  
        if epoch < self.warmup_epochs:
            lr = ((self.lr_max - self.lr_start) / self.warmup_epochs) * epoch + self.lr_start
        elif epoch <= (self.warmup_epochs + self.sustain_epochs):
            lr = self.lr_max
        else:
            progress = (
                (epoch - self.warmup_epochs - self.sustain_epochs) / 
                (self.epochs - self.warmup_epochs - self.sustain_epochs))
            lr = (self.lr_max-self.lr_min) * (0.5 * (1.0 + tf.math.cos((22/7) * 
                self.n_cycles * 2.0 * progress)))
            if self.lr_min is not None:
                lr = tf.math.maximum(self.lr_min, lr)
        return lr

    def plot(self):
        epochs = range(self.epochs+1)
        lr = [self(epoch) for epoch in epochs]
        plt.plot(epochs, lr)
        plt.xlabel("learning_rate")
        plt.ylabel("epochs")
        plt.show()

class CIEXYZNet(Model):
    def __init__(
            self, local_depth, local_convdepth, local_imagesize, global_depth, 
            global_convdepth, global_imagesize, scale):
        super(CIEXYZNet, self).__init__()
        self.local_depth = local_depth
        self.local_convdepth = local_convdepth
        self.local_imagesize = local_imagesize
        self.global_depth = global_depth
        self.global_convdepth = global_convdepth
        self.global_imagesize = global_imagesize
        self.global_output = 18
        self.scale = scale

        self.conv_params = {
            "strides": 1,
            "padding": "same",
            "kernel_initializer": "he_normal",
            "bias_initializer": "zeros"}        

        self.srgb2xyz_localnet = self.get_localsubnet(
            name="sRGB2XYZ_localnet", conv_depth=64, depth=self.local_depth)
        self.srgb2xyz_globalnet = self.get_globalsubnet(
            name="sRGB2XYZ_globalnet", 
            kernel_size=5,
            depth=6, conv_depth=64, 
            image_size=self.local_imagesize,
            output_nodes=2048,
            dropout=0.7)
        self.xyz2srgb_localnet = self.get_localsubnet(
            name="XYZ2sRGB_localnet", conv_depth=self.local_convdepth, depth=self.local_depth)        
        self.xyz2srgb_globalnet = self.get_globalsubnet(
            name="XYZ2sRGB_globalnet", 
            kernel_size=3,
            depth=self.global_depth, 
            conv_depth=self.global_convdepth, 
            image_size=self.global_imagesize, 
            output_nodes=512,
            dropout=0.3)

    def get_localsubnet(self, name, conv_depth, depth):
        local_subnet = Sequential(name=name)
               
        for i in range(depth):
            local_subnet.add(Conv2D(
                **self.conv_params,
                kernel_size=3,
                filters=conv_depth if i != depth - 1 else 3,
                activation=LeakyReLU() if i != depth - 1 else "tanh",
                name=f"conv_{i}"))

        local_subnet.add(Rescaling(self.scale))
        return local_subnet

    def get_globalsubnet(self, name, kernel_size, depth, conv_depth, image_size, output_nodes, dropout):
        global_subnet = Sequential(name=name)

        global_subnet.add(Resizing(
            height=image_size, 
            width=image_size,
            interpolation='bilinear'))

        for i in range(depth):
            global_subnet.add(Conv2D(
                **self.conv_params, 
                kernel_size=kernel_size,
                filters=conv_depth,
                activation=LeakyReLU(), 
                name=f"conv_{i}"))
            global_subnet.add(MaxPool2D(
                pool_size=2, strides=2, padding="valid", name=f"maxpool_{i}"))

        global_subnet.add(Flatten(name="flatten_0"))
        global_subnet.add(Dense(output_nodes, name="dense_0"))
        global_subnet.add(Dropout(dropout, name="dropout_0"))
        global_subnet.add(Dense(self.global_output, name="dense_1"))
        return global_subnet     

    def forward_local(self, x, target):
        if target == "xyz":
            x = self.srgb2xyz_localnet(x)
        elif target == "srgb":
            x = self.xyz2srgb_localnet(x)
        else:
            raise Exception("Wrong target")
        return x

    def transform(self, x):
        x = tf.reshape(x, (-1, 3))
        x_ = tf.identity(x)
        x_1 = []
        for _ in range(self.global_output // 9):
            x_ = tf.math.multiply(x, x_)
            x_1.append(x_)
        x = tf.concat(x_1, axis=1)
        return x

    def forward_global(self, x, target):
        if target == "xyz":
            m_v = self.srgb2xyz_globalnet(x)
        elif target == "srgb":
            m_v = self.xyz2srgb_globalnet(x)
        else:
            raise Exception("Wrong target")

        m_v = tf.reshape(m_v, (-1, self.global_output // 3, 3))

        y = []
        for i in range(len(m_v)):
            t1 = self.transform(x[i])
            t2 = m_v[i]
            t = tf.matmul(t1, t2)
            t = tf.clip_by_value(t, 0., 1.)
            t = tf.reshape(t, tf.shape(x)[1:])
            y.append(t)            
        y = tf.stack(y, axis=0)
        return y

    def call(self, x):
        # Encoder
        l_xyz = x - self.forward_local(x, target='xyz')
        xyz = self.forward_global(l_xyz, target='xyz')

        # Decoder
        g_srgb = self.forward_global(xyz, target='srgb')
        srgb = g_srgb + self.forward_local(g_srgb, target='srgb')
        return xyz, srgb