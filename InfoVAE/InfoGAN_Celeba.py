from __future__ import division
import os
import time
import math
from glob import glob
import scipy.io as sio
import tensorflow as tf
import numpy as np
from six.moves import xrange
from scipy.misc import imsave as ims
from HSICSupport import *
from ops import *
from Utlis2 import *
from NewExperiment.Common.Celeba_Hander import *
from keras.datasets import cifar10

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

discrete_len = 10
continous_len = 10
noise_len = 100

def file_name(file_dir):
    t1 = []
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "C:/commonData/rendered_chairs/rendered_chairs/" + a1 + "/renders/*.png"
            img_path = glob.glob(b1)
            t1.append(img_path)

        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return  cc

def encoder(image,dim_discrete,continous_dim, batch_size=64, reuse=False):
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, continous_len, 'e_mean')

        y_logits = linear(h5, dim_discrete, 'e_mix')
        y_softmax = tf.nn.softmax(y_logits)

        return y_softmax,y_logits,z_logits

def discriminator(image, batch_size=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.sigmoid(z_logits)
        return z_mix,z_logits

def generator(z_in,y_in, batch_size=64, reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()

        z = concat((z_in,y_in),axis=1)
        kernel = 3
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

def InfoGAN(x_hat,z_in,y_in):

    G = generator(z_in, y_in)

    D_real, D_real_logits = discriminator(x_hat)
    D_fake, D_fake_logits = discriminator(G,reuse=True)

    # get loss for discriminator
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

    d_loss = d_loss_real + d_loss_fake

    # get loss for generator
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

    ## 2. Information Loss
    code_fake, code_logit_fake, continous_fake = encoder(G,discrete_len,continous_len, reuse=False)
    # vae loss
    discrete_fake = code_logit_fake

    # discrete code : categorical
    disc_code_est = code_logit_fake
    disc_code_tg = y_in[:, :discrete_len]
    q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_code_est, labels=disc_code_tg))

    # continuous code : gaussian
    cont_code_est = continous_fake
    cont_code_tg = y_in[:, discrete_len:]
    q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))

    # get information loss
    q_loss = q_disc_loss + q_cont_loss

    T_vars = tf.trainable_variables()
    encoder_vars1 = [var for var in T_vars if var.name.startswith('encoder')]
    decoder_vars1 = [var for var in T_vars if var.name.startswith('decoder')]
    discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
    q_vars = encoder_vars1 + decoder_vars1 + discriminator_vars1

    learning_rate = 0.0002
    beta1 = 0.5
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
            .minimize(d_loss, var_list=discriminator_vars1)
        g_optim = tf.train.AdamOptimizer(learning_rate * 10, beta1=beta1) \
            .minimize(g_loss, var_list=decoder_vars1)
        q_optim = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1) \
            .minimize(q_loss, var_list=q_vars)

    return d_optim,d_loss,g_optim,g_loss,q_optim,q_loss,G

batch_size = 64

z_in = tf.placeholder(tf.float32, shape=[None, noise_len])
y_in = tf.placeholder(tf.float32, shape=[None, continous_len*2])
image_dims = [64, 64, 3]
x_hat = tf.placeholder(
    tf.float32, [batch_size] + image_dims, name='real_images')

d_optim,d_loss,g_optim,g_loss,q_optim,q_loss,Generated_Imags = InfoGAN(x_hat,z_in,y_in)

isWeight = False

ADD_NOISE = False
n_epochs = 9
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_valid = x_test[0:batch_size]

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if isWeight:
        saver.restore(sess, 'models/InfoGAN_Celeba')

        # load dataset
        img_path = glob.glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        n_examples = 202599
        total_batch = int(n_examples / batch_size)

        batch_files = data_files[0:
                                 batch_size]
        batch = [get_image(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)
        x_fixed = batch_images

        batch_labels = np.random.multinomial(1,
                                             discrete_len * [float(1.0 / discrete_len)],
                                             size=[batch_size])

        myArr = []
        for k in range(10):
            batch_labels = np.zeros((batch_size, discrete_len))
            batch_codes = np.concatenate((batch_labels, np.random.uniform(-1, 1, size=(batch_size, continous_len))),
                                         axis=1)
            batch_labels[:, k] = 1
            batch_z = np.random.uniform(-1, 1, [batch_size, noise_len]).astype(np.float32)

            y_PRR = sess.run(Generated_Imags, feed_dict={x_hat: batch_images, z_in: batch_z, y_in: batch_codes})
            y_RPR = np.reshape(y_PRR, (-1, 32, 32, 3))

            ims("results/" + "myh" + str(0) + ".png", merge2(y_RPR, [8, 8]))

            for k1 in range(8):
                myArr.append(y_RPR[k1])

        myArr = np.array(myArr)
        ims("results/" + "myh" + str(0) + ".png", merge2(myArr, [10, 8]))
        ims("results/" + "real" + str(0) + ".png", merge2(batch_images, [8, 8]))

        b1 = 0
        b = 0

        b = 0
    else:

        # load dataset
        img_path = glob.glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        n_examples = 202599
        total_batch = int(n_examples / batch_size)

        batch_files = data_files[0:
                                 batch_size]
        batch = [get_image(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)
        x_fixed = batch_images

        bestScore = 1000000

        batch_labels = np.random.multinomial(1,
                                             discrete_len * [float(1.0 / discrete_len)],
                                             size=[batch_size])
        batch_codes = np.concatenate((batch_labels, np.random.uniform(-1, 1, size=(batch_size, continous_len))),
                                     axis=1)
        batch_z = np.random.uniform(-1, 1, [batch_size, noise_len]).astype(np.float32)

        for epoch in range(n_epochs):
            count = 0
            # Random shuffling
            index = [i for i in range(np.shape(data_files)[0])]
            random.shuffle(index)
            data_files = data_files[index]

            # Loop over all batches
            for i in range(total_batch):
                batch_images = data_files[i * batch_size:
                                         (i + 1) * batch_size]

                batch = [get_image(
                    sample_file,
                    input_height=128,
                    input_width=128,
                    resize_height=64,
                    resize_width=64,
                    crop=True)
                    for sample_file in batch_images]

                batch_images = np.array(batch).astype(np.float32)
                # Compute the offset of the current minibatch in the data.
                batch_xs_input = batch_images
                batch_xs_target = batch_xs_input

                # add salt & pepper noise
                if ADD_NOISE:
                    batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                batch_labels = np.random.multinomial(1,
                                                     discrete_len * [float(1.0 / discrete_len)],
                                                     size=[batch_size])
                batch_codes = np.concatenate((batch_labels, np.random.uniform(-1, 1, size=(batch_size, continous_len))),
                                             axis=1)

                batch_z = np.random.uniform(-1, 1, [batch_size, noise_len]).astype(np.float32)

                # update D network
                _, d_loss_ = sess.run([d_optim, d_loss],
                                                       feed_dict={x_hat: batch_images, y_in: batch_codes,
                                                                  z_in: batch_z})
                # update G and Q network
                _, g_loss_, _, q_loss_ = sess.run([g_optim, g_loss, q_optim,q_loss],
                    feed_dict={x_hat: batch_images, z_in: batch_z, y_in: batch_codes})

                print("Epoch: [%2d], d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, d_loss_, g_loss_))

            # print cost every epoch

            y_PRR = sess.run(Generated_Imags, feed_dict={x_hat: batch_images, z_in: batch_z, y_in: batch_codes})
            y_RPR = np.reshape(y_PRR, (-1, 32, 32, 3))
            ims("results/" + "VAE" + str(epoch) + ".jpg", merge2(y_RPR[:64], [8, 8]))

            if epoch > 0:
                x_fixed_image = np.reshape(x_fixed, (-1, 32, 32, 3))
                ims("results/" + "Real" + str(epoch) + ".png", merge2(x_fixed_image[:64], [8, 8]))

            saver.save(sess, "models/InfoGAN_Celeba")
