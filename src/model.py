import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, LeakyReLU, Flatten, Dense, Dropout, Resizing, Rescaling)

from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

class CompiledMAELoss(Loss):
    def __init__(self, weight=1.5):
        super(CompiledMAELoss, self).__init__()
        self.weight = weight

    def call(self, y_true, y_pred):
        xyz_true, srgb_true = y_true
        xyz_pred, srgb_pred = y_pred
        xyz_loss = tf.abs(xyz_true - xyz_pred)
        srgb_loss = tf.abs(srgb_true - srgb_pred)
        total_loss = srgb_loss + (self.weight * xyz_loss)
        return total_loss

class CIEXYZNet(Model):
    def __init__(
            self, local_depth, local_convdepth, global_depth, 
            global_convdepth, global_imagesize, scale):
        super(CIEXYZNet, self).__init__()
        self.local_depth = local_depth
        self.local_convdepth = local_convdepth
        self.global_depth = global_depth
        self.global_convdepth = global_convdepth
        self.global_imagesize = global_imagesize
        self.scale = scale

        self.conv_params = {
            "kernel_size": 3,
            "strides": 1,
            "padding": "same",
            "kernel_initializer": "he_normal",
            "bias_initializer": "zeros"}        

        self.srgb2xyz_localnet = self.get_localsubnet(name="sRGB2XYZ_localnet")
        self.xyz2srgb_localnet = self.get_localsubnet(name="XYZ2sRGB_localnet")
        self.srgb2xyz_globalnet = self.get_globalsubnet(name="sRGB2XYZ_globalnet")
        self.xyz2srgb_globalnet = self.get_globalsubnet(name="XYZ2sRGB_globalnet")

    def get_localsubnet(self, name):
        local_subnet = Sequential(name=name)
               
        for i in range(self.local_depth):
            local_subnet.add(Conv2D(
                **self.conv_params,
                filters=self.local_convdepth if i != self.local_depth - 1 else 3,
                activation=LeakyReLU() if i != self.local_depth - 1 else "tanh",
                name=f"conv_{i}"))

        local_subnet.add(Rescaling(self.scale))
        return local_subnet

    def get_globalsubnet(self, name):
        global_subnet = Sequential(name=name)

        global_subnet.add(Resizing(
            height=self.global_imagesize, 
            width=self.global_imagesize,
            interpolation='bilinear'))

        for i in range(self.global_depth):
            global_subnet.add(Conv2D(
                **self.conv_params, 
                filters=self.global_convdepth,
                activation=LeakyReLU(), 
                name=f"conv_{i}"))
            global_subnet.add(MaxPool2D(
                pool_size=2, strides=2, padding="valid", name=f"maxpool_{i}"))

        global_subnet.add(Flatten(name="flatten_0"))
        global_subnet.add(Dense(1024, name="dense_0"))
        global_subnet.add(Dropout(0.5, name="dropout_0"))
        global_subnet.add(Dense(18, name="dense_1"))
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
        x = tf.reshape(x, (-1, 3)) # (65536, 3)
        x = tf.concat([x, tf.math.multiply(x, x)], axis=1) # (65536, 6)
        return x

    def forward_global(self, x, target):
        if target == "xyz":
            m_v = self.srgb2xyz_globalnet(x) # (8, 18)
        elif target == "srgb":
            m_v = self.xyz2srgb_globalnet(x) # (8, 18)
        else:
            raise Exception("Wrong target")

        m_v = tf.reshape(m_v, (x.shape[0], 6, 3)) # (8, 6, 3)

        y = []
        for i in range(m_v.shape[0]):
            t1 = self.transform(x[i]) # (65536, 6)
            t2 = m_v[i] # (6, 3)
            t = tf.matmul(t1, t2) # (65536, 3)
            t =  tf.reshape(t, x.shape[1:]) # (256, 256, 3)
            y.append(t)            
        return tf.stack(y, axis=0)
    
    def forward_srgb2xyz(self, srgb):
        l_xyz = srgb - self.forward_local(srgb, target='xyz')
        return self.forward_global(l_xyz, target='xyz')

    def forward_xyz2srgb(self, xyz):
        g_srgb = self.forward_global(xyz, target='srgb')
        return g_srgb + self.forward_local(g_srgb, target='srgb')

    def call(self, x):
        xyz = self.forward_srgb2xyz(x)
        srgb = self.forward_xyz2srgb(xyz)
        return xyz, srgb