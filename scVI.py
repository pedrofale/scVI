"""
Implementation of the single-cell Variational Inference (scVI) paper* without extensions
Pedro Ferreira

* Romain Lopez, Jeffrey Regier, Michael Cole, Michael Jordan, Nir Yosef, UC Berkeley

"""

import tensorflow as tf
from tensorflow.contrib import slim


def dense(x, num_outputs, activation=None, weight_init_std=0.01):
    """
    Defining the elementary layer of the network

    Variables:
    x: tensorflow variable
    num_outputs: number of outputs neurons after the dense layer
    activation: tensorflow activation function (tf.exp, tf.nn.relu...) for this layer
    """
    output = tf.identity(x)

    output = slim.fully_connected(output, num_outputs, activation_fn=activation,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=weight_init_std))

    return output


def gaussian_sample(mean, var, scope=None):
    """
    Function to sample from a multivariate gaussian with diagonal covariance in tensorflow

    Variables:
    mean: tf variable indicating the minibatch mean (shape minibatch_size x latent_space_dim)
    var: tf variable indicating the minibatch variance (shape minibatch_size x latent_space_dim)
    """

    with tf.variable_scope(scope, 'gaussian_sample'):
        sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
        sample.set_shape(mean.get_shape())
        return sample


def log_zinb_positive(x, mu, r, pi, eps=1e-8):
    """
    log likelihood (scalar) of a minibatch according to a ZINB model.
    we parametrize the Bernoulli using the logits, hence the softplus functions appearing

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    r: dispersion parameter (positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    case_zero = tf.nn.softplus(- pi + r * tf.log(r + eps) - r * tf.log(r + mu + eps)) - tf.nn.softplus(- pi)
    case_non_zero = - pi - tf.nn.softplus(- pi) + r * tf.log(r + eps) - r * tf.log(r + mu + eps) \
                    + x * tf.log(mu + eps) - x * tf.log(r + mu + eps) \
                    + tf.lgamma(x + r) - tf.lgamma(r) - tf.lgamma(x + 1)

    mask = tf.cast(tf.less(x, eps), tf.float32)
    res = tf.multiply(mask, case_zero) + tf.multiply(1 - mask, case_non_zero)

    return tf.reduce_sum(res, axis=-1)


class scVI(object):
    def __init__(self, n_input=100, n_layers=1, n_hidden=128, n_latent=2, weights_std_init=0.01, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.std = weights_std_init
        self.optimizer = optimizer

        self.build_model()  # build computation graph

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.n_input])

        self.inference_network()  # q(z|x)
        self.sampling_latent()
        self.generative_model()  # p(x|z)
        self.loss()  # update network weights

    def inference_network(self):
        """
        defines the inference network of the model, q(z|x)
        """

        h = dense(self.x, self.n_hidden, activation=tf.nn.relu)
        for layer in range(2, self.n_layers + 1):
            h = dense(h, self.n_hidden, activation=tf.nn.relu)

        self.qz_m = dense(h, self.n_latent, activation=None)  # variational distribution mean
        self.qz_v = dense(h, self.n_latent, activation=tf.exp)  # variational distribution variance (positive)

    def sampling_latent(self):
        """
        defines the sampling process on the latent space given the variational distribution
        """
        self.z = gaussian_sample(self.qz_m, self.qz_v)

    def generative_model(self):
        """
        defines the generative network of the model, p(x|z), with discrete variables integrated out
        there are two networks: one which outputs the Gamma means and one which outputs the Bernoulli logits
        """

        h = dense(self.z, self.n_hidden, activation=tf.nn.relu)

        for layer in range(2, self.n_layers + 1):
            h = dense(h, self.n_hidden, activation=tf.nn.relu)

        # mean of gamma distribution
        self.px_scale = dense(h, self.n_input, activation=tf.nn.softmax)

        # dispersion parameter of negative binomial or of gamma
        self.px_r = tf.Variable(tf.random_normal([self.n_input]), name="r")

        # mean of poisson (rate) is the mean of the gamma that parameterizes it
        self.px_rate = self.px_scale

        # dropout logit
        self.px_dropout = dense(h, self.n_input, activation=None)

    def loss(self):
        """
        write down the loss and the optimizer
        """
        # VAE loss
        recon = log_zinb_positive(self.x, self.px_rate, tf.exp(self.px_r), self.px_dropout)

        # KL divergence between two gaussians
        kl = 0.5 * tf.reduce_sum(tf.square(self.qz_m) + self.qz_v - tf.log(1e-8 + self.qz_v) - 1, 1)

        self.ELBO_gau = tf.reduce_mean(recon - kl)

        self.loss = - self.ELBO_gau

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = self.optimizer
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)
