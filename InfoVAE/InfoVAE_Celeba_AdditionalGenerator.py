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
continous_len = 100
noise_len = 100


def encoder(image, dim_discrete, continous_dim, batch_size=64, reuse=False):
    with tf.variable_scope("encoder") as scope:
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

        z_mean = linear(h5, continous_len, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        y_logits = linear(h5, dim_discrete, 'e_mix')
        y_softmax = tf.nn.softmax(y_logits)

        return y_softmax, y_logits, z_mean, z_log_sigma_sq


def discriminator(image, z_in, batch_size=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
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

        h4 = tf.concat((h4, z_in), axis=1)

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.sigmoid(z_logits)
        return z_mix, z_logits


def generator(z_in, y_in, batch_size=64, reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()

        z = concat((z_in, y_in), axis=1)
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

def generator2(z_in, y_in, batch_size=64, reuse=False):
    with tf.variable_scope("decoder2") as scope:
        if reuse:
            scope.reuse_variables()

        z = concat((z_in, y_in), axis=1)
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


def InfoVAE(x_hat, z_in, y_in):
    G = generator(z_in, y_in)

    code_real, code_logit_real, mean_real, std_real = encoder(x_hat, discrete_len, continous_len)
    discrete = code_logit_real
    continous = mean_real + std_real * tf.random_normal(tf.shape(mean_real), 0, 1, dtype=tf.float32)

    discrete_softmax = tf.nn.softmax(discrete) + 1e-10
    log_y = tf.log(discrete_softmax)
    discrete_real = my_gumbel_softmax_sample(log_y, np.arange(10))

    z_fake = tf.concat((discrete_real, continous), axis=1)
    z_real = y_in

    D_real, D_real_logits = discriminator(x_hat, z_fake)
    D_fake, D_fake_logits = discriminator(G, z_real, reuse=True)

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
    code_fake, code_logit_fake, mean_fake, std_fake = encoder(G, discrete_len, continous_len, reuse=True)

    # vae loss
    discrete_fake = code_logit_fake
    continous_fake = mean_fake + std_fake * tf.random_normal(tf.shape(mean_fake), 0, 1, dtype=tf.float32)

    discrete_softmax_fake = tf.nn.softmax(discrete_fake) + 1e-10
    log_y_fake = tf.log(discrete_softmax_fake)
    discrete_fake = my_gumbel_softmax_sample(log_y_fake, np.arange(10))

    z_fake2 = tf.concat((discrete_fake, continous_fake), axis=1)
    G_fake = generator(z_in, z_fake2, reuse=True)

    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(G - G_fake), [1, 2, 3]))

    KL_divergence1 = 0.5 * tf.reduce_sum(
        tf.square(mean_fake) + tf.square(std_fake) - tf.log(1e-8 + tf.square(std_fake)) - 1, 1)
    KL_divergence1 = tf.reduce_mean(KL_divergence1)

    # KL divergence on gumble softmax
    KL_y = tf.reduce_sum(code_fake * (tf.log(code_fake + 1e-10) - tf.log(1.0 / 10.0)), 1)
    KL_y = tf.reduce_mean(KL_y)

    vae_loss = reconstruction_loss + KL_divergence1 + 0.001 * KL_y

    # discrete code : categorical
    disc_code_est = code_logit_fake
    disc_code_tg = y_in[:, :discrete_len]
    q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_code_est, labels=disc_code_tg))

    # continuous code : gaussian
    cont_code_est = mean_fake + std_fake * tf.random_normal(tf.shape(mean_fake), 0, 1, dtype=tf.float32)
    cont_code_tg = y_in[:, discrete_len:]
    q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))

    # get information loss
    q_loss = q_disc_loss + q_cont_loss

    #VAE loss for standared vae
    G_real = generator2(discrete_real, continous, reuse=False)
    reconstruction_loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(x_hat - G_real), [1, 2, 3]))

    KL_divergence2 = 0.5 * tf.reduce_sum(
        tf.square(mean_real) + tf.square(std_real) - tf.log(1e-8 + tf.square(std_real)) - 1, 1)
    KL_divergence2 = tf.reduce_mean(KL_divergence2)

    # KL divergence on gumble softmax
    KL_y2 = tf.reduce_sum(code_real * (tf.log(code_real + 1e-10) - tf.log(1.0 / 10.0)), 1)
    KL_y2 = tf.reduce_mean(KL_y2)

    vae_loss2 = reconstruction_loss2 + KL_divergence2 + 0.001 * KL_y2


    T_vars = tf.trainable_variables()
    encoder_vars1 = [var for var in T_vars if var.name.startswith('encoder')]
    decoder_vars1 = [var for var in T_vars if var.name.startswith('decoder')]
    decoder_vars2 = [var for var in T_vars if var.name.startswith('decoder2')]
    discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
    q_vars = encoder_vars1 + decoder_vars1 + discriminator_vars1

    learning_rate = 0.0002
    beta1 = 0.5



    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
            .minimize(d_loss, var_list=discriminator_vars1)
        g_optim = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1) \
            .minimize(g_loss, var_list=decoder_vars1)
        q_optim = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1) \
            .minimize(q_loss, var_list=q_vars)
        vae_optim = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1) \
            .minimize(vae_loss, var_list=encoder_vars1)
        vae_stand = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1) \
            .minimize(vae_loss2, var_list=decoder_vars2)

    return d_optim, d_loss, g_optim, g_loss, q_optim, q_loss, vae_optim, vae_loss, G,vae_stand


def Give_GeneratedImages(z_in, y_in):
    G = generator(z_in, y_in, reuse=True)
    return G


def Give_Reconstructed(x_hat, z_in, y_in):
    code_fake, code_logit_fake, mean_fake, std_fake = encoder(x_hat, discrete_len, continous_len, reuse=True)
    discrete_fake = code_logit_fake
    continous_fake = mean_fake + std_fake * tf.random_normal(tf.shape(mean_fake), 0, 1, dtype=tf.float32)

    discrete_softmax_fake = tf.nn.softmax(discrete_fake) + 1e-10
    log_y_fake = tf.log(discrete_softmax_fake)
    discrete_fake = my_gumbel_softmax_sample(log_y_fake, np.arange(10))

    z_fake2 = tf.concat((discrete_fake, continous_fake), axis=1)
    G_fake = generator(z_in, z_fake2, reuse=True)
    return G_fake


batch_size = 64

z_in = tf.placeholder(tf.float32, shape=[None, noise_len])
y_in = tf.placeholder(tf.float32, shape=[None, continous_len + discrete_len])
image_dims = [64, 64, 3]
x_hat = tf.placeholder(
    tf.float32, [batch_size] + image_dims, name='real_images')

d_optim, d_loss, g_optim, g_loss, q_optim, q_loss, vae_optim, vae_loss, Generated_Imags,vae_stand = InfoVAE(x_hat, z_in, y_in)

myGenerated = Give_GeneratedImages(z_in, y_in)

myReconstructed = Give_Reconstructed(x_hat, z_in, y_in)

isWeight = False

ADD_NOISE = False
n_epochs = 10
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if isWeight:
        saver.restore(sess, 'models/InfoVAE_Celeba_AdditionalGenerator')

        '''
        import matplotlib.image as mpimg

        a1 = mpimg.imread('F:/Third_Experiment/results/myselection/InfoVAE_womanToMan0.png')
        a2 = mpimg.imread('F:/Third_Experiment/results/myselection/InfoVAE_womanToMan1.png')
        a3 = mpimg.imread('F:/Third_Experiment/results/myselection/InfoVAE_womanToMan2.png')
        a4 = mpimg.imread('F:/Third_Experiment/results/myselection/InfoVAE_womanToMan3.png')
        a5 = mpimg.imread('F:/Third_Experiment/results/myselection/InfoVAE_womanToMan4.png')

        myImage = np.zeros((5 * 64, 10 * 64, 3))
        myImage[0:64, 0:64 * 10, 0:3] = a1
        myImage[64:64 * 2, 0:64 * 10, 0:3] = a2
        myImage[64 * 2:64 * 3, 0:64 * 10, 0:3] = a3
        myImage[64 * 3:64 * 4, 0:64 * 10, 0:3] = a4
        myImage[64 * 4:64 * 5, 0:64 * 10, 0:3] = a5

        ims("results/" + "InfoVAE_Emotion" + str(00) + ".png", myImage)

        b = 0
        batch_labels = np.random.multinomial(1,
                                             discrete_len * [float(1.0 / discrete_len)],
                                             size=[batch_size])
        batch_z = np.random.uniform(-1, 1, [batch_size, noise_len]).astype(np.float32)

        continous_latents = np.random.uniform(-1, 1, size=(batch_size, continous_len))
        discrete_latents = batch_labels
        batch_codes = np.concatenate((discrete_latents, continous_latents), axis=1)

        minValue = -1.0
        changValue = 2.0 / 10.0
        myIndex = 0

        # load dataset
        img_path = glob.glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        n_examples = 202599
        total_batch = int(n_examples / batch_size)

        ttIndex = 100
        batch_files = data_files[ttIndex * batch_size:
                                 ttIndex * batch_size + batch_size]
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

        y_1 = sess.run(myGenerated, feed_dict={z_in: batch_z, x_hat: x_fixed, y_in: batch_codes})

        my111 = []
        for i in range(10):
            my111.append(y_1[i])
        for i in range(10):
            my111.append(x_fixed[i])

        my111 = np.array(my111)
        ims("results/" + "InfoVAE_" + str(0) + ".png", merge2(my111, [2, 10]))

        myNew1 = []
        myNew2 = []
        myNew3 = []
        myNew4 = []
        myNew5 = []

        for kk in range(64):
            myNews = []
            batch_labels = np.random.multinomial(1,
                                                 discrete_len * [float(1.0 / discrete_len)],
                                                 size=[batch_size])
            batch_z = np.random.uniform(-1, 1, [batch_size, noise_len]).astype(np.float32)

            continous_latents = np.random.uniform(-1, 1, size=(batch_size, continous_len))
            discrete_latents = batch_labels

            for i in range(10):
                # continous_latents[:,myIndex] = minValue + i * changValue
                batch_labels = np.zeros((batch_size, 10))
                if i % 2 == 0:
                    batch_labels[:, 0:5] = 1
                else:
                    batch_labels[:, 5:10] = 1

                batch_codes = np.concatenate((discrete_latents, continous_latents),
                                             axis=1)
                y_1 = sess.run(myGenerated, feed_dict={z_in: batch_z, y_in: batch_codes})
                myNews.append(y_1[0])

            myNews = np.array(myNews)
            ims("results/" + "InfoVAE_womanToMan" + str(kk) + ".png", merge2(myNews, [1, 10]))

        gc = 0
        '''
    else:
        saver.restore(sess, 'models/InfoVAE_Celeba_AdditionalGenerator')

        myIndex = 0
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
            index = [i for i in range(n_examples)]
            random.shuffle(index)
            data_files = data_files[index]

            # Loop over all batches
            for i in range(total_batch):
                batch_files = data_files[i * batch_size:
                                         (i + 1) * batch_size]
                batch = [get_image(
                    batch_file,
                    input_height=128,
                    input_width=128,
                    resize_height=64,
                    resize_width=64,
                    crop=True) \
                    for batch_file in batch_files]

                try:
                    batch_images = np.array(batch).astype(np.float32)
                except e:
                    print(e)

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

                myIndex = myIndex+1

                if myIndex > 15:
                # update D network
                    _, d_loss_ = sess.run([d_optim, d_loss],
                                          feed_dict={x_hat: batch_images, y_in: batch_codes,
                                                     z_in: batch_z})
                    myIndex = 0
                else:
                    d_loss_ = sess.run( d_loss,
                                          feed_dict={x_hat: batch_images, y_in: batch_codes,
                                                     z_in: batch_z})

                # update G and Q network
                _, g_loss_, _, q_loss_, _ = sess.run([g_optim, g_loss, q_optim, q_loss, vae_optim],
                                                     feed_dict={x_hat: batch_images, z_in: batch_z, y_in: batch_codes})

                #Train additional generator
                _ = sess.run(vae_stand,feed_dict={x_hat: batch_images, z_in: batch_z, y_in: batch_codes})

                print("Epoch: [%2d], d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, d_loss_, g_loss_))

            # print cost every epoch

            y_PRR = sess.run(Generated_Imags, feed_dict={x_hat: batch_images, z_in: batch_z, y_in: batch_codes})
            y_RPR = np.reshape(y_PRR, (-1, 64, 64, 3))
            ims("results/" + "VAE" + str(epoch) + ".jpg", merge2(y_RPR[:64], [8, 8]))

            if epoch > 0:
                x_fixed_image = np.reshape(x_fixed, (-1, 64, 64, 3))
                ims("results/" + "Real" + str(epoch) + ".png", merge2(x_fixed_image[:64], [8, 8]))

            saver.save(sess, "models/InfoVAE_Celeba_AdditionalGenerator")
