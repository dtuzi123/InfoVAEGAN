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
from tensorlayer.layers import *
from ops import *
from Utlis2 import *
from Support import *
import tensorlayer as tl
from Mixture_Models import *

distributions = tf.distributions


def custom_layer(input_matrix, mix, dropout, resue=False):
    # with tf.variable_scope("custom_layer",reuse=resue):
    # w_init = tf.contrib.layers.variance_scaling_initializer()
    # b_init = tf.constant_initializer(0.)

    # weights = tf.get_variable(name="mix_weights", initializer=[0.25,0.25,0.25,0.25],trainable=True)
    weights = mix
    a1 = input_matrix[:, 0, :] * dropout[:, 0:1]
    a2 = input_matrix[:, 1, :] * dropout[:, 1:2]
    a3 = input_matrix[:, 2, :] * dropout[:, 2:3]
    a4 = input_matrix[:, 3, :] * dropout[:, 3:4]
    a5 = input_matrix[:, 4, :] * dropout[:, 4:5]
    a6 = input_matrix[:, 5, :] * dropout[:, 5:6]

    w1 = mix[:, 0:1]
    w2 = mix[:, 1:2]
    w3 = mix[:, 2:3]
    w4 = mix[:, 3:4]
    w5 = mix[:, 4:5]
    w6 = mix[:, 5:6]

    outputs = w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4 + w5 * a5 + w6 * a6
    return outputs


def KL_Dropout2(log_alpha):
    ab = tf.cast(log_alpha, tf.float32)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695;
    C = -k1
    mdkl = k1 * tf.nn.sigmoid(k2 + k3 * ab) - 0.5 * tf.log1p(tf.exp(-ab)) + C
    return -tf.reduce_sum(mdkl)


def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, last_term):
    # encoding
    mu1, sigma1, mix1 = Create_Celeba_Encoder(x_hat, 64, "encoder1")
    mu2, sigma2, mix2 = Create_Celeba_Encoder(x_hat, 64, "encoder2")
    mu3, sigma3, mix3 = Create_Celeba_Encoder(x_hat, 64, "encoder3")
    mu4, sigma4, mix4 = Create_Celeba_Encoder(x_hat, 64, "encoder4")
    mu5, sigma5, mix5 = Create_Celeba_Encoder(x_hat, 64, "encoder5")
    mu6, sigma6, mix6 = Create_Celeba_Encoder(x_hat, 64, "encoder6")

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)
    z5 = distributions.Normal(loc=mu5, scale=sigma5)
    z6 = distributions.Normal(loc=mu6, scale=sigma6)

    init_min = 0.1
    init_max = 0.1
    init_min = (np.log(init_min) - np.log(1. - init_min))
    init_max = (np.log(init_max) - np.log(1. - init_max))
    dropout_a = tf.get_variable(name='dropout',
                                shape=None,
                                initializer=tf.random_uniform(
                                    (1,),
                                    init_min,
                                    init_max),
                                dtype=tf.float32,
                                trainable=True)
    dropout_p = tf.nn.sigmoid(dropout_a)

    dropout_b = 1 - dropout_p
    dropout_log = tf.log(dropout_p)
    dropout_log2 = tf.log(dropout_b)

    cats_range = np.zeros((batch_size * 6, 2))
    cats_range[:, 0] = 0
    cats_range[:, 1] = 1
    dropout_samples = gumbel_softmax_sample3(dropout_log, dropout_log2, cats_range, [batch_size * 6])
    dropout_samples = tf.reshape(dropout_samples, (-1, 6))

    dropout_regularizer = dropout_p * tf.log(dropout_p)
    dropout_regularizer += (1. - dropout_p) * tf.log(1. - dropout_p)
    dropout_regularizer *= dropout_regularizer * 10 * -1
    dropout_regularizer = tf.clip_by_value(dropout_regularizer, -10, 0)

    mix1 = mix1 * dropout_samples[:, 0:1]
    mix2 = mix2 * dropout_samples[:, 1:2]
    mix3 = mix3 * dropout_samples[:, 2:3]
    mix4 = mix4 * dropout_samples[:, 3:4]
    mix5 = mix5 * dropout_samples[:, 4:5]
    mix6 = mix6 * dropout_samples[:, 5:6]

    sum1 = mix1 + mix2 + mix3 + mix4 + mix5 + mix6
    mix1 = mix1 / sum1
    mix2 = mix2 / sum1
    mix3 = mix3 / sum1
    mix4 = mix4 / sum1
    mix5 = mix5 / sum1
    mix6 = mix6 / sum1

    sum1 = mix1 + mix2 + mix3 + mix4 + mix5 + mix6
    mix1 = mix1 / sum1
    mix2 = mix2 / sum1
    mix3 = mix3 / sum1
    mix4 = mix4 / sum1
    mix5 = mix5 / sum1
    mix6 = mix6 / sum1

    mix = tf.concat([mix1, mix2, mix3, mix4, mix5, mix6], 1)
    mix_parameters = mix
    dist = tf.distributions.Dirichlet(mix)
    mix_samples = dist.sample()
    mix = mix_samples

    # sampling by re-parameterization technique
    # z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
    z4_samples = z4.sample()
    z5_samples = z5.sample()
    z6_samples = z6.sample()

    ttf = []
    ttf.append(z1_samples)
    ttf.append(z2_samples)
    ttf.append(z3_samples)
    ttf.append(z4_samples)
    ttf.append(z5_samples)
    ttf.append(z6_samples)

    dHSIC_Value = dHSIC(ttf)

    # decoding
    y1 = Create_Celeba_SubDecoder_(z1_samples, 64, "decoder1")
    y2 = Create_Celeba_SubDecoder_(z2_samples, 64, "decoder2")
    y3 = Create_Celeba_SubDecoder_(z3_samples, 64, "decoder3")
    y4 = Create_Celeba_SubDecoder_(z4_samples, 64, "decoder4")
    y5 = Create_Celeba_SubDecoder_(z5_samples, 64, "decoder5")
    y6 = Create_Celeba_SubDecoder_(z6_samples, 64, "decoder6")

    y1 = tf.reshape(y1, (-1, 8 * 8 * 256))
    y2 = tf.reshape(y2, (-1, 8 * 8 * 256))
    y3 = tf.reshape(y3, (-1, 8 * 8 * 256))
    y4 = tf.reshape(y4, (-1, 8 * 8 * 256))
    y5 = tf.reshape(y5, (-1, 8 * 8 * 256))
    y6 = tf.reshape(y6, (-1, 8 * 8 * 256))

    # dropout out
    y1 = y1 * mix_samples[:, 0:1]
    y2 = y2 * mix_samples[:, 1:2]
    y3 = y3 * mix_samples[:, 2:3]
    y4 = y4 * mix_samples[:, 3:4]
    y5 = y5 * mix_samples[:, 4:5]
    y6 = y6 * mix_samples[:, 5:6]


    y1 = tf.reshape(y1, (batch_size, 8, 8, 256))
    y2 = tf.reshape(y2, (batch_size, 8, 8, 256))
    y3 = tf.reshape(y3, (batch_size, 8, 8, 256))
    y4 = tf.reshape(y4, (batch_size, 8, 8, 256))
    y5 = tf.reshape(y5, (batch_size, 8, 8, 256))
    y6 = tf.reshape(y6, (batch_size, 8, 8, 256))

    y = y1 + y2 + y3 + y4 + y5 + y6
    y = Create_Celeba_Generator_(y, 64, "final")

    m1 = np.zeros(dim_z, dtype=np.float32)
    m1[:] = 0
    v1 = np.zeros(dim_z, dtype=np.float32)
    v1[:] = 1

    # p_z1 = distributions.Normal(loc=np.zeros(dim_z, dtype=np.float32),
    #                           scale=np.ones(dim_z, dtype=np.float32))
    p_z1 = distributions.Normal(loc=m1,
                                scale=v1)

    m2 = np.zeros(dim_z, dtype=np.float32)
    m2[:] = 0
    v2 = np.zeros(dim_z, dtype=np.float32)
    v2[:] = 1
    p_z2 = distributions.Normal(loc=m2,
                                scale=v2)

    m3 = np.zeros(dim_z, dtype=np.float32)
    m3[:] = 0
    v3 = np.zeros(dim_z, dtype=np.float32)
    v3[:] = 1
    p_z3 = distributions.Normal(loc=m3,
                                scale=v3)

    m4 = np.zeros(dim_z, dtype=np.float32)
    m4[:] = 0
    v4 = np.zeros(dim_z, dtype=np.float32)
    v4[:] = 1
    p_z4 = distributions.Normal(loc=m4,
                                scale=v4)

    z = z1

    mu = mu1
    sigma = sigma1
    epsilon = 1e-8

    # additional loss
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - y), [1, 2, 3]))
    # kl_divergence = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + sigma - tf.square(mu) - tf.exp(sigma), 1))
    kl1 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z1, p_z1), 1))
    kl2 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, p_z2), 1))
    kl3 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z3, p_z3), 1))
    kl4 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z4, p_z4), 1))
    kl5 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z5, p_z4), 1))
    kl6 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z6, p_z4), 1))

    kl = kl1 + kl2 + kl3 + kl4 + kl5 + kl6
    kl_divergence = kl / 6.0

    # KL divergence between two Dirichlet distributions
    a1 = tf.clip_by_value(mix_parameters, 0.1, 0.8)
    a2 = tf.constant((0.17, 0.17, 0.17, 0.17, 0.17, 0.17), shape=(batch_size, 6))

    r = tf.reduce_sum((a1 - a2) * (tf.polygamma(0.0, a1) - tf.polygamma(0.0, 1)), axis=1)
    a = tf.lgamma(tf.reduce_sum(a1, axis=1)) - tf.lgamma(tf.reduce_sum(a2, axis=1)) + tf.reduce_sum(tf.lgamma(a2),
                                                                                                    axis=-1) - tf.reduce_sum(
        tf.lgamma(a1), axis=1) + r
    kl = a
    kl = tf.reduce_mean(kl)

    p1 = 1

    loss = reconstruction_loss + kl_divergence * p1 + kl + dHSIC_Value + dropout_regularizer
    KL_divergence = kl_divergence
    marginal_likelihood = reconstruction_loss

    return y, z, loss, -marginal_likelihood, kl_divergence,dropout_p,dropout_samples


def HiddenOutputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, last_term):
    mu1, sigma1, mix1 = Create_Celeba_Enoder(x_hat, 64, "encoder1", reuse=True)
    mu2, sigma2, mix2 = Create_Celeba_Encoder(x_hat, 64, "encoder2", reuse=True)
    mu3, sigma3, mix3 = Create_Celeba_Encoder(x_hat, 64, "encoder3", reuse=True)
    mu4, sigma4, mix4 = Create_Celeba_Encoder(x_hat, 64, "encoder4", reuse=True)
    mu5, sigma5, mix5 = Create_Celeba_Encoder(x_hat, 64, "encoder5", reuse=True)
    mu6, sigma6, mix6 = Create_Celeba_Encoder(x_hat, 64, "encoder6", reuse=True)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z1_samples = z1.sample()

    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z2_samples = z2.sample()
    c
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z3_samples = z3.sample()

    z4 = distributions.Normal(loc=mu4, scale=sigma4)
    z4_samples = z4.sample()

    z5 = distributions.Normal(loc=mu5, scale=sigma5)
    z5_samples = z5.sample()

    z6 = distributions.Normal(loc=mu6, scale=sigma6)
    z6_samples = z6.sample()

    return z1_samples, z2_samples, z3_samples, z4_samples, z5_samples, z6_samples


def Output_HiddenCode(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
    mu1, sigma1, mix1 = encoder(x_hat, batch_size=64, reuse=True)
    mu2, sigma2, mix2 = encoder2(x_hat, batch_size=64, reuse=True)
    mu3, sigma3, mix3 = encoder3(x_hat, batch_size=64, reuse=True)
    mu4, sigma4, mix4 = encoder4(x_hat, batch_size=64, reuse=True)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z1_samples = z1.sample()

    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z2_samples = z2.sample()

    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z3_samples = z3.sample()

    z4 = distributions.Normal(loc=mu4, scale=sigma4)
    z4_samples = z4.sample()

    sum1 = mix1 + mix2 + mix3 + mix4
    mix1 = mix1 / sum1
    mix2 = mix2 / sum1
    mix3 = mix3 / sum1
    mix4 = mix4 / sum1
    mix = tf.concat([mix1, mix2, mix3, mix4], 1)
    mix_parameters = mix
    dist = tf.distributions.Dirichlet(mix)
    mix_samples = dist.sample()
    mix = mix_samples

    return z1_samples, z2_samples, z3_samples, z4_samples, mix


n_hidden = 500
IMAGE_SIZE_MNIST = 28
dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image

myLatent_dim = 256
dim_z = myLatent_dim

# train
n_epochs = 5
batch_size = 64
learn_rate = 0.0001

# input placeholders

imagesize = 64
channel = 3
# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, imagesize, imagesize, channel], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, imagesize, imagesize, channel], name='input_img')

image_dims = [64, 64, 3]
x_hat = tf.placeholder(
    tf.float32, [batch_size] + image_dims, name='real_images')

x = x_hat
# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

last_term = tf.placeholder(tf.float32)

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence,dropout_p,dropout_samples = autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,
                                                                 last_term)
# z1_samples, z2_samples, z3_samples, z4_samples, z5_samples, z6_samples = HiddenOutputs(x_hat, x, dim_img, dim_z,
#                                                                                       n_hidden, keep_prob, last_term)

# optimization
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

# train

min_tot_loss = 1e99
ADD_NOISE = False

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内

isWeight = False
saver = tf.train.Saver(max_to_keep=4)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

    import glob

    if isWeight:
        saver.restore(sess, 'models/Dropout_Simple_Celeba6_Bernoulli')
        import glob

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
    lossArray = []
    NumberOfComponent = []
    dropoutArr = []
    MyCount = 0
    myIndex = 0
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

            _, tot_loss, loss_likelihood, loss_divergence,dropoutRate,dropoutSamples = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence,dropout_p,dropout_samples),
                feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 1.0})

            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, loss_likelihood, loss_divergence))
        # print cost every epoch

            MyCount = MyCount + 1
            if MyCount > 100:
                MyCount = 0
                myIndex = myIndex + 1

                dropoutArr.append(dropoutRate)
                lossArray.append(tot_loss)
                sum1 = 0
                for t2 in range(batch_size):
                    noCount = 0
                    for t1 in range(6):
                        if dropoutSamples[t2, t1] < 0.1:
                            noCount = noCount + 1
                    yesCount = 6 - noCount
                    sum1 = sum1 + yesCount

                sum1 = float(sum1 / batch_size)
                NumberOfComponent.append(sum1)

            if myIndex == 100:
                # save data
                lossArr1 = np.array(lossArray).astype('str')
                f = open("results/Celeba_lossArr.txt", "w", encoding="utf-8")
                for i in range(np.shape(lossArr1)[0]):
                    f.writelines(lossArr1[i])
                    f.writelines('\n')
                f.flush()
                f.close()

                lossArr1 = np.array(NumberOfComponent).astype('str')
                f = open("results/Celeba_Components.txt", "w", encoding="utf-8")
                for i in range(np.shape(lossArr1)[0]):
                    f.writelines(lossArr1[i])
                    f.writelines('\n')
                f.flush()
                f.close()

                lossArr1 = np.array(dropoutArr).astype('str')
                f = open("results/Celeba_DropoutRate.txt", "w", encoding="utf-8")
                for i in range(np.shape(lossArr1)[0]):
                    f.writelines(lossArr1[i])
                    f.writelines('\n')
                f.flush()
                f.close()

            print(myIndex)

        y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1})
        y_RPR = np.reshape(y_PRR, (-1, 64, 64, 3))
        ims("results/" + "VAE" + str(epoch) + ".jpg", merge2(y_RPR[:64], [8, 8]))

        loss_likelihood = loss_likelihood * -1
        if bestScore > loss_likelihood:
            bestScore = loss_likelihood
            #saver.save(sess, "models/Dropout_Simple_Celeba6_Bernoulli")

        if epoch > 0:
            x_fixed_image = np.reshape(x_fixed, (-1, 64, 64, 3))
            ims("results/" + "Real" + str(epoch) + ".jpg", merge2(x_fixed_image[:64], [8, 8]))

    # saver.save(sess, "F:/Third_Experiment/Dropout_Simple_Celeba4")

    # save data
    lossArr1 = np.array(lossArray).astype('str')
    f = open("results/Celeba_lossArr.txt", "w", encoding="utf-8")
    for i in range(np.shape(lossArr1)[0]):
        f.writelines(lossArr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    lossArr1 = np.array(NumberOfComponent).astype('str')
    f = open("results/Celeba_Components.txt", "w", encoding="utf-8")
    for i in range(np.shape(lossArr1)[0]):
        f.writelines(lossArr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    lossArr1 = np.array(dropoutArr).astype('str')
    f = open("results/Celeba_DropoutRate.txt", "w", encoding="utf-8")
    for i in range(np.shape(lossArr1)[0]):
        f.writelines(lossArr1[i])
        f.writelines('\n')
    f.flush()
    f.close()