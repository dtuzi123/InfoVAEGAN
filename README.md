# InfoVAEGAN
This is the implementation of the InfoVAEGAN. This project also includes the implementation of the Deep Mixture Generative Autoencoder (Please see more details in https://github.com/dtuzi123/Mixture-of-VAE-with-dropout).

Title : InfoVAEGAN: learning joint interpretable representations by information maximization

# Paper link

https://www.sciencedirect.com/science/article/pii/S0020025521002449

# Abstract

Learning disentangled and interpretable representations is an important task in
deep learning. Methods based on variational autoencoders (VAEs) generally yield
unclear and blurred images when comparing with other powerful generative models
such as Generative Adversarial Networks (GANs). In this paper, we propose a
novel hybrid model based on VAEs and GANs, namely InfoVAEGAN, a technique
aiming for learning both discrete and continuous interpretable representations in
an unsupervised manner. We achieve this by introducing the maximization of the
mutual information between joint latent variables and those created through the
generative processes. In order to learn an accurate inference network that can
infer exact interpretable representations, we introduce a lower bound on the loglikelihood
of the generator distribution and maximize it by using stochastic gradient
decent with the reparameterization trick. Experimental results performed on a
variety of datasets demonstrate that InfoVAEGAN is able to discover interpretable
and disentangled data representations. Moreover, InfoVAEGAN is able to generate
high quality images when setting parameters specific to the discrete and continuous
spaces.

# Environment

1. Tensorflow 1.5
2. Python 3.6

# How to run?

It notes that the file name "InfoVAE" is our InfoVAEGAN model. You can directly run the file by python such as python InfoVAE_3DChairs.py.


# BibTex

@article{ye2021learning,
  title={Learning joint latent representations based on information maximization},
  author={Ye, Fei and Bors, Adrian G},
  journal={Information Sciences},
  volume={567},
  pages={216--236},
  year={2021},
  publisher={Elsevier}
}

# How the difference between InfoVAEGAN and other related works

![image](https://github.com/dtuzi123/InfoVAEGAN/blob/master/a1.PNG)

# Visual results

![image](https://github.com/dtuzi123/InfoVAEGAN/blob/master/a2.PNG)

![image](https://github.com/dtuzi123/InfoVAEGAN/blob/master/a3.PNG)

![image](https://github.com/dtuzi123/InfoVAEGAN/blob/master/a4.PNG)

![image](https://github.com/dtuzi123/InfoVAEGAN/blob/master/a5.PNG)

![image](https://github.com/dtuzi123/InfoVAEGAN/blob/master/a6.PNG)








