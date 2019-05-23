import tensorflow as tf
import mnist_data

import tensorflow.contrib.slim as slim
import time
import seaborn as sns
from utils import *
from scipy.misc import imsave as ims
from Assign_Dataset import *
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from Support import *
from Mnist_DataHandle import *
from HSICSupport import *


import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from Utlis2 import *
from Support import *
from tensorlayer.layers import *
import tensorlayer as tl
from ops import *
import prettytensor as pt

distributions = tf.distributions

d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')
'''
e_bn2 = batch_norm(name='e_bn2')
e_bn3 = batch_norm(name='e_bn3')
e_bn4 = batch_norm(name='e_bn4')
'''
g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')

def Create_Encoder_MNIST(x, n_hidden, n_output, keep_prob, scopeName,resue=False):
    with tf.variable_scope(scopeName, reuse=resue):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2 + 1], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2 + 1], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:n_output * 2])
        mix = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output * 2:n_output * 2 + 1])
        return mean, stddev, mix

def Create_Encoder_MNIST_Conditional(x,y, n_hidden, n_output, keep_prob, scopeName,resue=False):
    with tf.variable_scope(scopeName, reuse=resue):
        dim_y = int(y.get_shape()[1])
        input = tf.concat(axis=1, values=[x, y])

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden+dim_y], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden+dim_y], initializer=b_init)
        h0 = tf.matmul(input, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2 + 1], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2 + 1], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:n_output*2])
        mix = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output * 2:n_output * 2 + 1])
    return mean, stddev, mix

def Create_FinalDecoder_Conditional(z, n_hidden, n_output, keep_prob,scopeName, reuse=False):
    with tf.variable_scope(scopeName, reuse=reuse):

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [z.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(z, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

def Create_Decoder_MNIST(z, n_hidden, n_output, keep_prob, scopeName,reuse=False):
    with tf.variable_scope(scopeName, reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

def Create_SubDecoder_Conditional(z,y, n_hidden, n_output, keep_prob, scopeName,reuse=False):
    with tf.variable_scope(scopeName, reuse=reuse):
        input = tf.concat(axis=1, values=[z, y])
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(input, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)
        return h0

def Create_SubDecoder(z, n_hidden, n_output, keep_prob, scopeName,reuse=False):
    with tf.variable_scope(scopeName, reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)
        return h0

def Create_FinalDecoder(z, n_hidden, n_output, keep_prob, scopeName,reuse=False):
    with tf.variable_scope(scopeName, reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [z.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(z, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

def Create_Generator(z,scoreName, batch_size=64, reuse=False):
    with tf.variable_scope(scoreName) as scope:
        if reuse:
            scope.reuse_variables()

        is_train = True
        image_size = 64  # 64 the output size of generator
        s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(
            image_size / 16)  # 32,16,8,4
        gf_dim = 64
        c_dim = 3 # n_color 3
        batch_size = batch_size # 64

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(z, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim * 4 * s8 * s8, W_init=w_init,
                            act=tf.identity, name='g/h0/lin')
        # net_h0.outputs._shape = (b_size,256*8*8)
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s8, s8, gf_dim * 4], name='g/h0/reshape')
        # net_h0.outputs._shape = (b_size,8,8,256)
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h0/batch_norm')

        # upsampling
        net_h1 = DeConv2d(net_h0, gf_dim * 4, (5, 5), strides=(2, 2),
                          padding='SAME', act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h1/batch_norm')
        # net_h1.outputs._shape = (b_size,16,16,256)

        net_h2 = DeConv2d(net_h1, gf_dim * 2, (5, 5), strides=(2, 2),
                          padding='SAME', act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h2/batch_norm')
        # net_h2.outputs._shape = (b_size,32,32,128)

        net_h3 = DeConv2d(net_h2, gf_dim // 2, (5, 5),  strides=(2, 2),
                          padding='SAME', act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='g/h3/batch_norm')
        # net_h3.outputs._shape = (b_size,64,64,32)

        # no BN on last deconv
        net_h4 = DeConv2d(net_h3, c_dim, (5, 5),  strides=(1, 1),
                          padding='SAME', act=None, W_init=w_init, name='g/h4/decon2d')
        # net_h4.outputs._shape = (b_size,64,64,3)
        # net_h4 = Conv2d(net_h3, c_dim, (5,5),(1,1), padding='SAME', W_init=w_init, name='g/h4/conv2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return logits

def Create_Encoder_Celeba(image,scopeName ,batch_size=64, reuse=False):
    z_dim = 128  # 512
    ef_dim = 64  # encoder filter number

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    is_train = True

    with tf.variable_scope(scopeName, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(image, name='en/in')  # (b_size,64,64,3)
        net_h0 = Conv2d(net_in, ef_dim, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='en/h0/conv2d')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='en/h0/batch_norm')
        # net_h0.outputs._shape = (b_size,32,32,64)

        net_h1 = Conv2d(net_h0, ef_dim * 2, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='en/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='en/h1/batch_norm')
        # net_h1.outputs._shape = (b_size,16,16,64*2)

        net_h2 = Conv2d(net_h1, ef_dim * 4, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='en/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='en/h2/batch_norm')
        # net_h2.outputs._shape = (b_size,8,8,64*4)

        net_h3 = Conv2d(net_h2, ef_dim * 8, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='en/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='en/h3/batch_norm')

        # mean of z
        net_h4 = FlattenLayer(net_h3)
            # net_h4.outputs._shape = (b_size,8*8*64*4)
        net_out1 = DenseLayer(net_h4, n_units=z_dim, act=tf.identity,
                                  W_init=w_init)
        net_out1 = BatchNormLayer(net_out1, act=tf.identity,
                                      is_train=is_train, gamma_init=gamma_init)

            # net_out1 = DenseLayer(net_h4, n_units=z_dim, act=tf.nn.relu,
            #         W_init = w_init, name='en/h4/lin_sigmoid')
        z_mean = net_out1.outputs  # (b_size,512)

            # log of variance of z(covariance matrix is diagonal)
        net_h5 = FlattenLayer(net_h3)
        net_out2 = DenseLayer(net_h5, n_units=z_dim, act=tf.identity,
                                  W_init=w_init,name='en/h4/lin_sigmoid')
        net_out2 = BatchNormLayer(net_out2, act=tf.nn.softplus,
                                      is_train=is_train, gamma_init=gamma_init,name='en/out2/batch_norm')
            # net_out2 = DenseLayer(net_h5, n_units=z_dim, act=tf.nn.relu,
            #         W_init = w_init, name='en/h5/lin_sigmoid')
            # log of variance of z(covariance matrix is diagonal)
        net_h6 = FlattenLayer(net_h3, name='en/h5/flatten')
        net_out6 = DenseLayer(net_h6, n_units=1, act=tf.identity,
                                  W_init=w_init, name='en/h4/lin_sigmoid4')
        net_out6 = BatchNormLayer(net_out6, act=tf.nn.softplus,
                                      is_train=is_train, gamma_init=gamma_init,name='en/h4/lin_sigmoid6')
            # net_out2 = DenseLayer(net_h5, n_units=z_dim, act=tf.nn.relu,
            #         W_init = w_init, name='en/h5/lin_sigmoid')
        z_log_sigma_sq = net_out2.outputs + 1e-6  # (b_size,512)
        mix = net_out6.outputs + 1e-6  # (b_size,512)

    return z_mean, z_log_sigma_sq,mix

def Create_Celeba_SubDecoder(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        return h1

def Create_Celeba_SubDecoder_(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        return h1


def Create_Celeba_Generator(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        #h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        #h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        #h1 = tf.nn.relu(g_bn1(h1))
        h1 = z

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Create_Celeba_Generator_(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        #h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        #h1 = tf.reshape(z, [batch_size, 8, 8, 256])
        #h1 = tf.nn.relu(g_bn1(h1))
        h1 = z

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Create_Celeba_finalGenerator(z, batch_size=64,scopeName='bb', reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        kernel  = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Create_Celeba_Encoder(image, batch_size=64, scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)
        return (z_mean, z_log_sigma_sq,z_mix)

def Create_Celeba_final_Encoder(image, batch_size=64, scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = 3
        z_dim = 256
        h1 = image
        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)
        return (z_mean, z_log_sigma_sq, z_mix)

def Create_Celeba_Sub_Encoder(image, batch_size=64, scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        return h1


def Create_Celeba_Discriminator(image, batch_size=64, scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        is_training = True
        x = image
        kernel = 3
        # 经过这一步卷积后，(64,28,28,1)-->(64,14,14,64)
        net = lrelu(conv2d(x, 64, kernel, kernel, 2, 2, name='d_conv1'))
        # 经过这一步卷积后，(64,14,14,64)-->(64,7,7,128)
        net = lrelu(
            bn(conv2d(net, 128, kernel, kernel, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))  # 数据标准化

        net = lrelu(
            bn(conv2d(net, 256, kernel, kernel, 2, 2, name='d_conv3'), is_training=is_training,
               scope='d_bn3'))  # 数据标准化

        net = lrelu(
            bn(conv2d(net, 512, kernel, kernel, 2, 2, name='d_conv4'), is_training=is_training,
               scope='d_bn4'))  # 数据标准化

        net = tf.reshape(net, [batch_size, -1])
        # 经过线性处理后将矩阵，(64,1024)-->(64,1)
        out_logit = linear(net, 1, scope='d_fc4')
        # 将数据处理在（0,1）之间
        out = tf.nn.sigmoid(out_logit)

        return out, out_logit


def Create_CIFAR10_Encoder(input_data,dim_z,scopeName,reuse = False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        is_training = True
        batch_size = 64
        net = lrelu(conv2d(input_data, 64, 5, 5, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 5, 5, 2, 2, name='d_conv2'), is_training=is_training,
                       scope='d_bn2'))
        net = lrelu(bn(conv2d(net, 256, 5, 5, 2, 2, name='d_conv3'), is_training=is_training,
                       scope='d_bn3'))
        net = lrelu(bn(conv2d(net, 512, 5, 5, 2, 2, name='d_conv4'), is_training=is_training,
                       scope='d_bn4'))
        net = tf.reshape(net, [batch_size, -1])

        z_mean = linear(net, dim_z, 'e_mean')
        z_log_sigma_sq = linear(net, dim_z, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(net, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)

        return (z_mean, z_log_sigma_sq, z_mix)

def Create_CIFAR10_SubDecoder_(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()
        h_size_16 = 2
        is_training = True

        net = linear(z, 512 * h_size_16 * h_size_16, scope='g_fc1')
        net = tf.nn.relu(
            bn(tf.reshape(net, [batch_size, h_size_16, h_size_16, 512]), is_training=is_training, scope='g_bn1')
        )
        return net

def Create_CIFAR10_FinalDecoder_(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        h_size = 32
        h_size_2 = 16
        h_size_4 = 8
        h_size_8 = 4
        h_size_16 = 2
        is_training = True
        output_height = 32
        output_width = 32
        c_dim = 3

        '''
        net = linear(z, 512 * h_size_16 * h_size_16, scope='g_fc1')
        net = tf.nn.relu(
            bn(tf.reshape(net, [batch_size, h_size_16, h_size_16, 512]), is_training=is_training, scope='g_bn1')
        )
        '''
        net = z
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, h_size_8, h_size_8, 256], 5, 5, 2, 2,
                        name='g_dc2'), is_training=is_training, scope='g_bn2')
        )
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, h_size_4, h_size_4, 128], 5, 5, 2, 2,
                        name='g_dc3'), is_training=is_training, scope='g_bn3')
        )
        net = tf.nn.relu(
            bn(deconv2d(net, [batch_size, h_size_2, h_size_2, 64], 5, 5, 2, 2,
                        name='g_dc4'), is_training=is_training, scope='g_bn4')
        )
        out = tf.nn.tanh(
            deconv2d(net, [batch_size, output_height, output_width, c_dim], 5, 5, 2, 2,
                     name='g_dc5')
        )
        return out


def Create_CIFAR10_Encoder2(input_data,dim_z,scopeName,reuse = False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(input_data, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)
        return (z_mean, z_log_sigma_sq, z_mix)

def Create_CIFAR10_SubDecoder_2(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope(scopeName) as scope:
        if reuse:
            scope.reuse_variables()
        h_size_16 = 2
        is_training = True

        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        return h1

def Create_CIFAR10_FinalDecoder_2(z,batch_size=64,scopeName="d",reuse=False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h1 = z
        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))
        '''
        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [batch_size, 32, 32, 64],
                      kernel, kernel, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8