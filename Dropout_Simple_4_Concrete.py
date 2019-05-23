import tensorflow as tf
import mnist_data

import tensorflow.contrib.slim as slim
import time
import seaborn as sns
from Assign_Dataset import *
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from Support import *
from Mnist_DataHandle import *
from HSICSupport import *
from scipy.misc import imsave as ims
from utils import *

import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

distributions = tf.distributions
from Mixture_Models import *


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        j = int(j)

        img[int(j * h):int(j * h + h), int(i * w):int(i * w + w)] = image

    return img


def custom_layer(input_matrix, mix, resue=False):
    # with tf.variable_scope("custom_layer",reuse=resue):
    # w_init = tf.contrib.layers.variance_scaling_initializer()
    # b_init = tf.constant_initializer(0.)

    # weights = tf.get_variable(name="mix_weights", initializer=[0.25,0.25,0.25,0.25],trainable=True)
    weights = mix
    a1 = input_matrix[:, 0, :]
    a2 = input_matrix[:, 1, :]
    a3 = input_matrix[:, 2, :]
    a4 = input_matrix[:, 3, :]
    a5 = input_matrix[:, 4, :]
    a6 = input_matrix[:, 5, :]
    a7 = input_matrix[:, 6, :]
    a8 = input_matrix[:, 7, :]
    a9 = input_matrix[:, 8, :]
    a10 = input_matrix[:, 9, :]

    w1 = mix[:, 0:1]
    w2 = mix[:, 1:2]
    w3 = mix[:, 2:3]
    w4 = mix[:, 3:4]
    w5 = mix[:, 4:5]
    w6 = mix[:, 5:6]
    w7 = mix[:, 6:7]
    w8 = mix[:, 7:8]
    w9 = mix[:, 8:9]
    w10 = mix[:, 9:10]

    outputs = w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4 + w5 * a5 + w6 * a6 + w7 * a7 + w8 * a8 + w9 * a9 + w10 * a10
    return outputs


def KL_Dropout2(log_alpha):
    ab = tf.cast(log_alpha, tf.float32)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695;
    C = -k1
    mdkl = k1 * tf.nn.sigmoid(k2 + k3 * ab) - 0.5 * tf.log1p(tf.exp(-ab)) + C
    return -tf.reduce_sum(mdkl)


# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, last_term, dropout_in):
    # encoding
    mu1, sigma1, mix1 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder1")
    mu2, sigma2, mix2 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder2")
    mu3, sigma3, mix3 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder3")
    mu4, sigma4, mix4 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder4")

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    init_min = 0.1
    init_max = 0.1
    init_min = (np.log(init_min) - np.log(1. - init_min))
    init_max = (np.log(init_max) - np.log(1. - init_max))

    dropout_a =  tf.get_variable(name='dropout',
                 shape=None,
                 initializer=tf.random_uniform(
                     (1,),
                     init_min,
                     init_max),
                 dtype=tf.float32,
                 trainable=True)

    dropout_p = tf.nn.sigmoid(dropout_a[0])

    eps = 1e-7
    temp = 0.1

    unif_noise = tf.random_uniform(shape=[batch_size,4])
    drop_prob = (
            tf.log(dropout_p + eps)
            - tf.log(1. - dropout_p + eps)
            + tf.log(unif_noise + eps)
            - tf.log(1. - unif_noise + eps)
    )
    drop_prob = tf.nn.sigmoid(drop_prob / temp)
    random_tensor = 1. - drop_prob
    retain_prob = 1. - dropout_p
    dropout_samples = random_tensor / retain_prob

    dropout_regularizer = dropout_p * tf.log(dropout_p)
    dropout_regularizer += (1. - dropout_p) * tf.log(1. - dropout_p)
    dropout_regularizer *= dropout_regularizer *10 * -1
    dropout_regularizer = tf.clip_by_value(dropout_regularizer,-10,0)

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

    mix_dropout1 = dropout_samples[:,0:1]*mix_samples[:,0:1]
    mix_dropout2 = dropout_samples[:,1:2]*mix_samples[:,1:2]
    mix_dropout3 = dropout_samples[:,2:3]*mix_samples[:,2:3]
    mix_dropout4 = dropout_samples[:,3:4]*mix_samples[:,3:4]

    sum1 = mix_dropout1+mix_dropout2+mix_dropout3+mix_dropout4
    mix_dropout1 = mix_dropout1/sum1
    mix_dropout2 = mix_dropout2/sum1
    mix_dropout3 = mix_dropout3/sum1
    mix_dropout4 = mix_dropout4/sum1

    # sampling by re-parameterization technique
    # z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
    z4_samples = z4.sample()

    ttf = []
    ttf.append(z1_samples)
    ttf.append(z2_samples)
    ttf.append(z3_samples)
    ttf.append(z4_samples)

    dHSIC_Value = dHSIC(ttf)

    # decoding
    y1 = Create_SubDecoder(z1_samples, n_hidden, dim_img, keep_prob, "decoder1")
    y2 = Create_SubDecoder(z2_samples, n_hidden, dim_img, keep_prob, "decoder2")
    y3 = Create_SubDecoder(z3_samples, n_hidden, dim_img, keep_prob, "decoder3")
    y4 = Create_SubDecoder(z4_samples, n_hidden, dim_img, keep_prob, "decoder4")

    # dropout out
    y1 = y1 * mix_dropout1
    y2 = y2 * mix_dropout2
    y3 = y3 * mix_dropout3
    y4 = y4 * mix_dropout4

    y = y1 + y2 + y3 + y4
    output = Create_FinalDecoder(y, n_hidden, dim_img, keep_prob, "final")
    y = output

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

    kl1 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z1, p_z1), 1))
    kl2 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, p_z2), 1))
    kl3 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z3, p_z3), 1))
    kl4 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z4, p_z4), 1))

    KL_divergence = (kl1 + kl2 + kl3 + kl4) / 4.0

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)

    # KL divergence between two Dirichlet distributions
    a1 = tf.clip_by_value(mix_parameters, 0.1, 0.8)
    a2 = tf.constant((0.25, 0.25, 0.25, 0.25), shape=(batch_size, 4))
    
    r = tf.reduce_sum((a1 - a2) * (tf.polygamma(0.0, a1) - tf.polygamma(0.0, 1)), axis=1)
    a = tf.lgamma(tf.reduce_sum(a1, axis=1)) - tf.lgamma(tf.reduce_sum(a2, axis=1)) + tf.reduce_sum(tf.lgamma(a2),
                                                                                                    axis=-1) - tf.reduce_sum(
        tf.lgamma(a1), axis=1) + r
    kl = a
    kl = tf.reduce_mean(kl)

    p1 = 1
    p2 = 1
    p4 = 1
    ELBO = marginal_likelihood - KL_divergence * p2

    loss = -ELBO + kl * p1 + p4 * dHSIC_Value + dropout_regularizer

    z = z1_samples
    return y, z, loss, -marginal_likelihood, dropout_regularizer,dropout_p,dropout_samples


def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y


def HiddenOuputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, last_term):
    # encoding
    mu1, sigma1, mix1 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder1", True)
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2, mix2 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder2", True)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3, mix3 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder3", True)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4, mix4 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder4", True)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
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

def Final_Output(z1_samples, z2_samples, z3_samples, z4_samples,mix_samples, dim_img, dim_z, n_hidden, keep_prob):
    ard_init = -10.
    with tf.variable_scope("", reuse=True):  # root variable scope
        dropout_a = tf.get_variable("dropout")

    # Dropout of components
    m1 = np.ones(batch_size)
    s1 = np.zeros(batch_size)
    dropout_a = tf.cast(dropout_a, tf.float64)

    dropout_dis = distributions.Bernoulli(logits=None, probs=dropout_a)
    dropout_samples = dropout_dis.sample(sample_shape=(batch_size, 4))
    dropout_samples = tf.reshape(dropout_samples, (batch_size, 4))
    dropout_samples = tf.cast(dropout_samples, tf.float32)

    # decoding
    y1 = Create_SubDecoder(z1_samples, n_hidden, dim_img, keep_prob, "decoder1", True)
    y2 = Create_SubDecoder(z2_samples, n_hidden, dim_img, keep_prob, "decoder2", True)
    y3 = Create_SubDecoder(z3_samples, n_hidden, dim_img, keep_prob, "decoder3", True)
    y4 = Create_SubDecoder(z4_samples, n_hidden, dim_img, keep_prob, "decoder4", True)

    mix_dropout1 = dropout_samples[:,0:1]*mix_samples[:,0:1]
    mix_dropout2 = dropout_samples[:,1:2]*mix_samples[:,1:2]
    mix_dropout3 = dropout_samples[:,2:3]*mix_samples[:,2:3]
    mix_dropout4 = dropout_samples[:,3:4]*mix_samples[:,3:4]

    sum1 = mix_dropout1+mix_dropout2+mix_dropout3+mix_dropout4
    mix_dropout1 = mix_dropout1/sum1
    mix_dropout2 = mix_dropout2/sum1
    mix_dropout3 = mix_dropout3/sum1
    mix_dropout4 = mix_dropout4/sum1

    # dropout out
    y1 = y1 * mix_dropout1
    y2 = y2 * mix_dropout2
    y3 = y3 * mix_dropout3
    y4 = y4 * mix_dropout4

    y = y1 + y2 + y3 + y4
    output = Create_FinalDecoder(y, n_hidden, dim_img, keep_prob, "final", True)
    output = tf.clip_by_value(output, 1e-8, 1 - 1e-8)
    y = output
    return y

def CalculateMSE(x1,x2):
    c = tf.square(x1 - x2)
    return  tf.reduce_mean(c)

n_hidden = 500
IMAGE_SIZE_MNIST = 28
dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image

dim_z = 50

# train
n_epochs = 100
batch_size = 128
learn_rate = 0.001

train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
n_samples = train_size
# input placeholders

# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
dropout_in = tf.placeholder(tf.float32, shape=[batch_size,4], name='dropout_variables')

last_term = tf.placeholder(tf.float32, shape=[10])
Component_Count = tf.placeholder(tf.float32)

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence,dropoutA,dropoutSamples = autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,
                                                                 last_term, dropout_in)
z1_samples, z2_samples, z3_samples, z4_samples, mix = HiddenOuputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,
                                                                   last_term)
# y1, y2, y3, y4 = Outpiuts_Component(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

# optimization
t_vars = tf.trainable_variables()
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, var_list=t_vars)

# train

total_batch = int(n_samples / batch_size)

min_tot_loss = 1e99
ADD_NOISE = False

train_data2_ = train_total_data[:, :-mnist_data.NUM_LABELS]
train_y = train_total_data[:, 784:784 + mnist_data.NUM_LABELS]

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_fixed = train_data2_[0:128]
saver = tf.train.Saver()

isWeight = False

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})
    lastvalue = np.ones((4))

    if isWeight:
        saver.restore(sess, 'models/Dropout_Sinmple_4_Bernoulli')
        '''
        x_train = np.reshape(x_train, (-1, 28 * 28))
        colorStr = ['blue', 'red', 'black', 'cyan', 'darkcyan', 'darkred', 'green', 'hotpink', 'lightblue', 'magenta']
        marks = ['o', 'v', '^', '<', '+', '>', 'x', 'h', 'H', 's']
        import matplotlib.pyplot as plt

        for i in range(10):
            x_l = x_train[i * batch_size:i * batch_size + batch_size]
            y_l = y_train[i * batch_size:i * batch_size + batch_size]
            z1_samples_, z2_samples_, z3_samples_, z4_samples_, mix_ = sess.run(
                (z1_samples, z2_samples, z3_samples, z4_samples, mix),
                feed_dict={x_hat: x_l, x: x_l, keep_prob: 0.9})


            z1_samples_ = z1_samples_ * mix_[:, 0:1]
            z2_samples_ = z2_samples_ * mix_[:, 1:2]
            z3_samples_ = z3_samples_ * mix_[:, 2:3]
            z4_samples_ = z4_samples_ * mix_[:, 3:4]


            for i in range(np.shape(z1_samples_)[0]):
                index = y_l[i]
                # plt.scatter(z1_samples_[i, 0], z1_samples_[i, 1],color=colorStr[index])

                plt.scatter(z1_samples_[i,0], z1_samples_[i,1],color=colorStr[index],marker=marks[index])
                # plt.scatter(z2_samples_[i,0], z2_samples_[i,1],color=colorStr[index],marker=marks[index])
                # plt.scatter(z3_samples_[i, 0], z3_samples_[i, 1],color=colorStr[index],marker=marks[index])
                #plt.scatter(z4_samples_[i, 0], z4_samples_[i, 1], color=colorStr[index], marker=marks[index])

        plt.show()
        b = 0
        '''

        tIndex = 7
        x_fixed = x_train[batch_size * tIndex:batch_size * tIndex + batch_size]
        x_fixed = x_test[batch_size * tIndex:batch_size * tIndex + batch_size]
        x_fixed = np.reshape(x_fixed, (-1, 28 * 28))

        reco,dropoutA,dropoutSamples = sess.run((y,dropoutA,dropoutSamples), feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})
        reco = np.reshape(reco, (-1, 28, 28))
        x_fixed = np.reshape(x_fixed,(-1,28,28))
        ims("results/" + "MNIST__Real_Bernoulli" + str(0) + ".png", merge(x_fixed[:64], [8, 8]))
        ims("results/" + "MNIST__Reco_Bernoulli" + str(0) + ".png", merge(reco[:64], [8, 8]))

        # yy = sess.run(y,feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})
        # yy = np.reshape(yy,(-1,28,28))
        # ims("results/" + "VAE" + str(0) + ".jpg", merge(yy[:64], [8, 8]))

        z_in1 = tf.placeholder(tf.float32, shape=[None, dim_z])
        z_in2 = tf.placeholder(tf.float32, shape=[None, dim_z])
        z_in3 = tf.placeholder(tf.float32, shape=[None, dim_z])
        z_in4 = tf.placeholder(tf.float32, shape=[None, dim_z])
        mix_in = tf.placeholder(tf.float32, shape=[None, 4])
        x_in1 = tf.placeholder(tf.float32, shape=[None, 28*28])
        x_in2 = tf.placeholder(tf.float32, shape=[None, 28*28])

        mse = CalculateMSE(x_in1,x_in2)

        yy = Final_Output(z_in1, z_in2, z_in3, z_in4,mix_in, dim_img, dim_z, n_hidden, keep_prob)

        x_fixed = np.reshape(x_fixed, (-1, 28 * 28))
        z1_samples_, z2_samples_, z3_samples_, z4_samples_, mix_ = sess.run(
            (z1_samples, z2_samples, z3_samples, z4_samples, mix),
            feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})

        mix_[:,:] = 0
        mix_[:,0:1] = 1
        y1 = sess.run(yy, feed_dict={z_in1: z1_samples_, z_in2: z2_samples_, z_in3: z3_samples_, z_in4: z4_samples_,
                                     keep_prob: 1.0, mix_in: mix_})
        y1 = np.reshape(y1,(-1,28,28))
        ims("results/" + "AAA" + str(0) + ".png", merge(y1[:64], [8, 8]))

        tCount = 0
        mySamples = dropoutSamples[0]
        for i in range(4):
            if mySamples[i] == 1:
                tCount = tCount+1

        y1 = np.reshape(y1,(-1,28*28))
        reco = np.reshape(reco,(-1,28*28))
        mse1 = sess.run(mse,feed_dict={x_in1:x_fixed,x_in2:reco})
        mse2 = sess.run(mse,feed_dict={x_in1:x_fixed,x_in2:y1})
        mse1 = mse1 * 28*28
        mse2 = mse2 * 28*28
        myCount = tCount
        bc = 0

        x_fixed = np.reshape(x_fixed,(-1,28*28))
        z1_samples_, z2_samples_, z3_samples_, z4_samples_, mix_ = sess.run(
            (z1_samples, z2_samples, z3_samples, z4_samples, mix),
            feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})

        z2_samples_ = np.zeros((batch_size, dim_z))
        z3_samples_ = np.zeros((batch_size, dim_z))
        z4_samples_ = np.zeros((batch_size, dim_z))
        y1 = sess.run(yy, feed_dict={z_in1: z1_samples_, z_in2: z2_samples_, z_in3: z3_samples_, z_in4: z4_samples_,
                                     keep_prob: 1.0,mix_in:mix_})

        z1_samples_, z2_samples_, z3_samples_, z4_samples_, mix_ = sess.run(
            (z1_samples, z2_samples, z3_samples, z4_samples, mix),
            feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})

        z1_samples_ = np.zeros((batch_size, dim_z))
        z3_samples_ = np.zeros((batch_size, dim_z))
        z4_samples_ = np.zeros((batch_size, dim_z))
        y2 = sess.run(yy, feed_dict={z_in1: z1_samples_, z_in2: z2_samples_, z_in3: z3_samples_, z_in4: z4_samples_,
                                     keep_prob: 1.0,mix_in:mix_})

        z1_samples_, z2_samples_, z3_samples_, z4_samples_, mix_ = sess.run(
            (z1_samples, z2_samples, z3_samples, z4_samples, mix),
            feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})

        z2_samples_ = np.zeros((batch_size, dim_z))
        z1_samples_ = np.zeros((batch_size, dim_z))
        z4_samples_ = np.zeros((batch_size, dim_z))
        y3 = sess.run(yy, feed_dict={z_in1: z1_samples_, z_in2: z2_samples_, z_in3: z3_samples_, z_in4: z4_samples_,
                                     keep_prob: 1.0,mix_in:mix_})

        z1_samples_, z2_samples_, z3_samples_, z4_samples_, mix_ = sess.run(
            (z1_samples, z2_samples, z3_samples, z4_samples, mix),
            feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})

        z2_samples_ = np.zeros((batch_size, dim_z))
        z3_samples_ = np.zeros((batch_size, dim_z))
        z1_samples_ = np.zeros((batch_size, dim_z))
        y4 = sess.run(yy, feed_dict={z_in1: z1_samples_, z_in2: z2_samples_, z_in3: z3_samples_, z_in4: z4_samples_,
                                     keep_prob: 1.0,mix_in:mix_})

        z1_samples_, z2_samples_, z3_samples_, z4_samples_, mix_ = sess.run(
            (z1_samples, z2_samples, z3_samples, z4_samples, mix),
            feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})

        y5 = sess.run(y, feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 1.0})

        yyArr = np.zeros((batch_size, 28 * 28))
        yyArr[0:20, :] = x_fixed[0:20]
        yyArr[20:40, :] = y1[0:20]
        yyArr[40:60, :] = y2[0:20]
        yyArr[60:80, :] = y3[0:20]
        yyArr[80:100, :] = y4[0:20]
        yyArr[100:120, :] = y5[0:20]

        yyArr = np.reshape(yyArr, (-1, 28, 28))

        ims("results/" + "MNIST_Components" + str(0) + ".png", merge(yyArr[:100], [5, 20]))
        bc = 0

    valid_reco = Give_Reconstruction(x_hat, x)
    x_valid = x_test[0:batch_size]
    x_valid = np.reshape(x_valid, (-1, 28 * 28))
    bestScore = 10000000
    x_train = np.reshape(x_train, (-1, 28 * 28))

    n_epochs = 100
    dropout_count = batch_size * 4
    for epoch in range(n_epochs):
        # Random shuffling
        # Random shuffling
        index = [i for i in range(np.shape(x_train)[0])]
        random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]
        x_fixed = x_train[0:batch_size]
        x_fixed = np.reshape(x_fixed,(-1,28*28))

        # Loop over all batches
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            # Compute the offset of the current minibatch in the data.
            batch_xs_input = x_train[i * batch_size:i * batch_size + batch_size]
            batch_xs_target = batch_xs_input

            if ADD_NOISE:
                batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)


            dropout1 = sess.run(dropoutA)

            _, tot_loss, loss_likelihood, loss_divergence,dropoutSamples_ = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence,dropoutSamples),
                feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 1.0})

        y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1.0})
        y_RPR = np.reshape(y_PRR, (-1, 28, 28))
        ims("results/" + "VAE" + str(epoch) + ".jpg", merge(y_RPR[:64], [8, 8]))

        print(dropout1)
        print("epoch %f: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
            epoch, tot_loss, loss_likelihood, loss_divergence))

        if epoch > 0:
            x_fixed_image = np.reshape(x_fixed, (-1, 28, 28))
            ims("results/" + "Real" + str(epoch) + ".jpg", merge(x_fixed_image[:64], [8, 8]))

        y_PRR = sess.run(y, feed_dict={x_hat: x_valid, keep_prob: 1})
        recoValue = sess.run(valid_reco, feed_dict={x_hat: x_valid, x: y_PRR})
        if bestScore < recoValue:
            bestScore = recoValue
            # saver.save(sess, 'models/Dropout_Sinmple_4_Bernoulli')

    saver.save(sess, 'models/Dropout_Sinmple_4_Bernoulli')
