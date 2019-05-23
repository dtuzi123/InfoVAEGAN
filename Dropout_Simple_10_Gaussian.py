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
from utils import *

import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

distributions = tf.distributions
from Mixture_Models import *

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

    outputs = w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4 + w5 * a5+w6 * a6+w7 * a7+w8 * a8+w9 * a9+w10 * a10
    return outputs

# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, last_term,Component_Count):
    # encoding
    mu1, sigma1, mix1 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder1")
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2, mix2 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder2")
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3, mix3 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder3")
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4, mix4 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder4")
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    mu5, sigma5, mix5 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder5")
    mu6, sigma6, mix6 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder6")
    mu7, sigma7, mix7 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder7")
    mu8, sigma8, mix8 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder8")
    mu9, sigma9, mix9 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder9")
    mu10, sigma10, mix10 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob,"encoder10")

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)
    z5 = distributions.Normal(loc=mu5, scale=sigma5)
    z6 = distributions.Normal(loc=mu6, scale=sigma6)
    z7 = distributions.Normal(loc=mu7, scale=sigma7)
    z8 = distributions.Normal(loc=mu8, scale=sigma8)
    z9 = distributions.Normal(loc=mu9, scale=sigma9)
    z10 = distributions.Normal(loc=mu10, scale=sigma10)

    p = 0.5
    #a = p / (1.0-p)
    ard_init = -10.
    dropout_a = tf.get_variable("dropout",shape=[1],initializer=tf.constant_initializer(ard_init))

    #Dropout of components
    m1 = np.ones(batch_size)
    s1 = np.zeros(batch_size)
    dropout_a = tf.cast(dropout_a,tf.float64)

    dropout_dis = distributions.Normal(loc=m1, scale=dropout_a)
    dropout_samples = dropout_dis.sample(sample_shape=(10))
    dropout_samples = tf.transpose(dropout_samples)
    dropout_samples = tf.cast(dropout_samples, tf.float32)
    dropout_samples = tf.clip_by_value(dropout_samples, 1e-8, 1 - 1e-8)

    mix1 = dropout_samples[:,0:1]
    mix2 = dropout_samples[:,1:2]
    mix3 = dropout_samples[:,2:3]
    mix4 = dropout_samples[:,3:4]
    mix5 = dropout_samples[:,4:5]
    mix6 = dropout_samples[:,5:6]
    mix7 = dropout_samples[:,6:7]
    mix8 = dropout_samples[:,7:8]
    mix9 = dropout_samples[:,8:9]
    mix10 = dropout_samples[:,9:10]

    sum1 = mix1 + mix2 + mix3 + mix4 + mix5 +mix6+mix7+mix8+mix9+mix10
    mix1 = mix1 / sum1
    mix2 = mix2 / sum1
    mix3 = mix3 / sum1
    mix4 = mix4 / sum1
    mix5 = mix5 / sum1
    mix6 = mix6 / sum1
    mix7 = mix7 / sum1
    mix8 = mix8 / sum1
    mix9 = mix9 / sum1
    mix10 = mix10 / sum1

    mix = tf.concat([mix1, mix2, mix3, mix4,mix5,mix6,mix7,mix8,mix9,mix10], 1)
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
    z7_samples = z7.sample()
    z8_samples = z8.sample()
    z9_samples = z9.sample()
    z10_samples = z10.sample()

    ttf = []
    ttf.append(z1_samples)
    ttf.append(z2_samples)
    ttf.append(z3_samples)
    ttf.append(z4_samples)
    ttf.append(z5_samples)
    ttf.append(z6_samples)
    ttf.append(z7_samples)
    ttf.append(z8_samples)
    ttf.append(z9_samples)
    ttf.append(z10_samples)

    '''
    h1 = hsic_individual(z1_samples,z2_samples)
    h2 = hsic_individual(z1_samples,z3_samples)
    h3 = hsic_individual(z1_samples,z4_samples)
    h4 = hsic_individual(z2_samples,z3_samples)
    h5 = hsic_individual(z2_samples,z4_samples)
    h6 = hsic_individual(z3_samples,z4_samples)
    dHSIC_Value = h1+h2+h3+h4+h5+h6
    '''
    dHSIC_Value = dHSIC(ttf)
    #dHSIC_Value = last_term

    # decoding
    y1 = Create_SubDecoder(z1_samples, n_hidden, dim_img, keep_prob,"decoder1")
    #y1 = tf.clip_by_value(y1, 1e-8, 1 - 1e-8)

    y2 = Create_SubDecoder(z2_samples, n_hidden, dim_img, keep_prob, "decoder2")
    #y2 = tf.clip_by_value(y2, 1e-8, 1 - 1e-8)

    y3 = Create_SubDecoder(z3_samples, n_hidden, dim_img, keep_prob, "decoder3")
    #y3 = tf.clip_by_value(y3, 1e-8, 1 - 1e-8)

    y4 = Create_SubDecoder(z4_samples, n_hidden, dim_img, keep_prob, "decoder4")
    #y4 = tf.clip_by_value(y4, 1e-8, 1 - 1e-8)

    y5 = Create_SubDecoder(z5_samples, n_hidden, dim_img, keep_prob, "decoder5")
    #y5 = tf.clip_by_value(y5, 1e-8, 1 - 1e-8)

    y6 = Create_SubDecoder(z6_samples, n_hidden, dim_img, keep_prob, "decoder6")
    #y6 = tf.clip_by_value(y6, 1e-8, 1 - 1e-8)

    y7 = Create_SubDecoder(z7_samples, n_hidden, dim_img, keep_prob, "decoder7")
    #y7 = tf.clip_by_value(y7, 1e-8, 1 - 1e-8)

    y8 = Create_SubDecoder(z8_samples, n_hidden, dim_img, keep_prob, "decoder8")
    #y8 = tf.clip_by_value(y8, 1e-8, 1 - 1e-8)

    y9 = Create_SubDecoder(z9_samples, n_hidden, dim_img, keep_prob, "decoder9")
    #y9 = tf.clip_by_value(y9, 1e-8, 1 - 1e-8)

    y10 = Create_SubDecoder(z10_samples, n_hidden, dim_img, keep_prob, "decoder10")
    #y10 = tf.clip_by_value(y10, 1e-8, 1 - 1e-8)

    #dropout out
    y1 = y1 * mix_samples[:, 0:1]
    y2 = y2 * mix_samples[:, 1:2]
    y3 = y3 * mix_samples[:, 2:3]
    y4 = y4 * mix_samples[:, 3:4]
    y5 = y5 * mix_samples[:, 4:5]
    y6 = y6 * mix_samples[:, 5:6]
    y7 = y7 * mix_samples[:, 6:7]
    y8 = y8 * mix_samples[:, 7:8]
    y9 = y9 * mix_samples[:, 8:9]
    y10 = y10 * mix_samples[:, 9:10]

    y = y1 + y2+y3+y4+y5+y6+y7+y8+y9+y10
    output = Create_FinalDecoder(y, n_hidden, dim_img, keep_prob, "final", reuse=False)
    #output = tf.clip_by_value(output, 1e-8, 1 - 1e-8)
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
    kl5 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z5, p_z4), 1))
    kl6 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z6, p_z4), 1))
    kl7 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z7, p_z4), 1))
    kl8 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z8, p_z4), 1))
    kl9 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z9, p_z4), 1))
    kl10 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z10, p_z4), 1))

    kl1 = kl1
    kl2 = kl2
    kl3 = kl3
    kl4 = kl4
    kl5 = kl5
    kl6 = kl6
    kl7 = kl7
    kl8 = kl8
    kl9 = kl9
    kl10 = kl10

    KL_divergence = (kl1 + kl2 + kl3 + kl4 + kl5+kl6+kl7+kl8+kl9+kl10) / 10.0

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)

    # KL divergence between two Dirichlet distributions
    a1 = mix_parameters
    a2 = tf.constant((0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1), shape=(batch_size, 10))

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

    loss = -ELBO + kl * p1 + p4*dHSIC_Value #diverse_KL_divergence

    z = z1_samples
    return y, z, loss, -marginal_likelihood, KL_divergence


def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y

def HiddenOuputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, last_term):
    # encoding
    mu1, sigma1, mix1 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder1",True)
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2, mix2 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder2",True)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3, mix3 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder3",True)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4, mix4 = Create_Encoder_MNIST(x_hat, n_hidden, dim_z, keep_prob, "encoder4",True)
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

    return z1_samples, z2_samples, z3_samples, z4_samples,mix


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
last_term = tf.placeholder(tf.float32,shape=[10])
Component_Count = tf.placeholder(tf.float32)

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence = autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,
                                                                 last_term,Component_Count)
z1_samples, z2_samples, z3_samples, z4_samples,mix = HiddenOuputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, last_term)
#y1, y2, y3, y4 = Outpiuts_Component(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

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

isWeight = True

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})
    lastvalue = np.ones((4))

    if isWeight:
        saver.restore(sess, 'models/Dropout_Sinmple_10_Gaussian')
        x_fixed = np.reshape(x_fixed,(-1,28*28))
        z1_samples, z2_samples, z3_samples, z4_samples,mix = sess.run((z1_samples, z2_samples, z3_samples, z4_samples,mix),feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 0.9})

        '''
        z1_samples = z1_samples * mix[:,0:1]
        z2_samples = z2_samples * mix[:,1:2]
        z3_samples = z3_samples * mix[:,2:3]
        z4_samples = z4_samples * mix[:,3:4]
        '''

        z_samples = z1_samples + z2_samples+z3_samples + z4_samples
        z_in2 = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
        yy = Create_Decoder_MNIST(z_in2, n_hidden, dim_img, keep_prob,'decoder1',True)

        z_samples = z1_samples+z2_samples
        y1 = sess.run(yy,feed_dict={z_in2:z_samples, keep_prob: 0.9})

        z_samples = z2_samples
        y2 = sess.run(yy, feed_dict={z_in2: z_samples, keep_prob: 0.9})

        z_samples = z3_samples
        y3 = sess.run(yy, feed_dict={z_in2: z_samples, keep_prob: 0.9})

        z_samples = z4_samples
        y4 = sess.run(yy, feed_dict={z_in2: z_samples, keep_prob: 0.9})

        z_samples = z1_samples+z2_samples+z3_samples+z4_samples
        y5 = sess.run(yy, feed_dict={z_in2: z_samples, keep_prob: 0.9})

        yyArr = np.zeros((batch_size,28*28))
        yyArr[0:10,:] = x_fixed[0:10]
        yyArr[10:20,:] = y1[0:10]
        yyArr[20:30,:] = y2[0:10]
        yyArr[30:40,:] = y3[0:10]
        yyArr[40:50,:] = y4[0:10]
        yyArr[50:60,:] = y5[0:10]

        yyArr = np.reshape(yyArr,(-1,28,28))

        ims("results/" + "VAE" + str(0) + ".jpg", merge(yyArr[:60], [6, 10]))
        bc = 0
    else:
        lastvalue = np.ones((10))
        valid_reco = Give_Reconstruction(x_hat,x)
        x_valid = x_test[0:batch_size]
        x_valid = np.reshape(x_valid,(-1,28*28))
        bestScore = 100000

        n_epochs = 100
        for epoch in range(n_epochs):
            # Random shuffling
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_xs_target = batch_xs_input

                if ADD_NOISE:
                    batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 1.0})

            y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1,last_term:lastvalue})
            y_RPR = np.reshape(y_PRR, (-1, 28, 28))
            ims("results/" + "VAE" + str(epoch) + ".jpg", merge(y_RPR[:64], [8, 8]))

            print("epoch %f: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, loss_likelihood, loss_divergence))

            if epoch > 0:
                x_fixed_image = np.reshape(x_fixed, (-1, 28, 28))
                ims("results/" + "Real" + str(epoch) + ".jpg", merge(x_fixed_image[:64], [8, 8]))

            y_PRR = sess.run(y, feed_dict={x_hat: x_valid, keep_prob: 1})
            recoValue = sess.run(valid_reco, feed_dict={x_hat: x_valid, x: y_PRR})
            if bestScore < recoValue:
                bestScore = recoValue
                saver.save(sess, 'models/Dropout_Sinmple_10_Gaussian')