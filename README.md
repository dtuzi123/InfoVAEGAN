# InfoVAEGAN
This is the implementation of the InfoVAEGAN

Title : InfoVAEGAN: learning joint interpretable representations by information maximization

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
