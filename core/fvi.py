import gpflowSlim as gfs
import tensorflow as tf

from core.grad_estimator import SpectralScoreEstimator, entropy_surrogate
import os.path as osp
import os
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.neural_kernel import AbstractNeuralKernel


class AbstractFVI(object):
    """
    Base Class for Functional Variational Inference.

    :param posterior: the posterior network to be optimized.
    :param rand_generator: Generates measurement points.
    :param obs_var: Float. Observation variance.
    :param input_dim. Int.
    :param n_rand. Int. Number of random measurement points.
    :param injected_noise: Float. Injected to function outputs for stability.
    """
    def __init__(self, posterior, rand_generator, obs_var, input_dim, 
                    n_rand, injected_noise, likelihood=None, classification=False,
                    num_classes=None):
        self.classification = classification
        self.num_classes = num_classes
        self.posterior = posterior
        self._rand_generator = rand_generator
        self.obs_var = obs_var
        self._likelihood = likelihood

        self.input_dim = input_dim
        self.n_rand = n_rand
        self.injected_noise = injected_noise

        self.init_inputs()
        self.build_rand()
        self.build_function()
        self.build_log_likelihood()
        self.build_evaluation()

    def init_inputs(self):
        tf_type = tf.float32
        self.x                = tf.placeholder(tf_type, shape=[None, self.input_dim], name='x')
        self.x_pred           = tf.placeholder(tf_type, shape=[None, self.input_dim], name='x_pred')
        if self.classification:
            self.y            = tf.placeholder(tf_type, shape=[None, self.num_classes], name='y')
        else:
            self.y            = tf.placeholder(tf_type, shape=[None], name='y')
        self.n_particles      = tf.placeholder(tf.int32, shape=[], name='n_particles')
        self.learning_rate_ph = tf.placeholder(tf_type, shape=[], name='learning_rate')

        self.coeff_ll         = tf.placeholder(tf_type, shape=[], name='coeff_ll')
        self.coeff_kl         = tf.placeholder(tf_type, shape=[], name='coeff_kl')

    @property
    def batch_size(self):
        return tf.to_float(tf.shape(self.x)[0])

    def build_rand(self):
        self.rand = self._rand_generator(self)
        self.x_rand = tf.concat([self.x, self.rand], axis=0)

    def build_function(self):
        self.repeat_x_rand = tf.tile(tf.expand_dims(self.x_rand, 0), [self.n_particles, 1, 1])

        # [n_particles, batch_size + n_rand]
        self.func_x_rand = self.posterior(self.x_rand, self.n_particles)
        self.func_x = self.func_x_rand[:, :tf.shape(self.x)[0]]
        self.func_x_pred = self.posterior(self.x_pred, self.n_particles)

        self.noisy_func_x_rand = self.func_x_rand + self.injected_noise * tf.random_normal(shape=tf.shape(self.func_x_rand))

    def build_log_likelihood(self):
        if self.classification:
            y_obs = tf.tile(tf.expand_dims(self.y, axis=0), [self.n_particles, 1, 1])
        else:
            y_obs = tf.tile(tf.expand_dims(self.y, axis=0), [self.n_particles, 1])
        # y_x_dist = tf.distributions.Normal(self.func_x, tf.to_float(self.obs_var)**0.5)
        # self.log_likelihood_sample = y_x_dist.log_prob(y_obs)
        # self.log_likelihood = tf.reduce_mean(self.log_likelihood_sample)
        # self.y_x_pred = y_x_dist.sample()

        if self._likelihood is not None:
            if self.classification:
                self.log_likelihood_sample, self.log_likelihood = self._likelihood.forward(self.func_x, y_obs)
            else:
                self.log_likelihood = tf.reduce_mean(self._likelihood.forward(self.func_x, y_obs))
                self.log_likelihood_sample = self._likelihood.forward(self.func_x, y_obs)
        else:
            y_x_dist = tf.distributions.Normal(self.func_x, tf.to_float(self.obs_var)**0.5)
            self.log_likelihood_sample = y_x_dist.log_prob(y_obs)
            self.log_likelihood = tf.reduce_mean(self.log_likelihood_sample)
            self.y_x_pred = y_x_dist.sample() ## not in original??

    def build_evaluation(self):
        if self.classification:
            preds = tf.argmax(tf.reduce_mean(self.func_x, 0), axis=-1, output_type=tf.int32)
            y = tf.argmax(self.y, axis=-1, output_type=tf.int32)
            equality = tf.equal(preds, y)
            self.eval_accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        else:
            self.eval_rmse = tf.sqrt(tf.reduce_mean((tf.reduce_mean(self.func_x, 0) - self.y) ** 2))
        self.eval_lld = tf.reduce_mean(tf.reduce_logsumexp(self.log_likelihood_sample, 0)
                                       - tf.log(tf.to_float(self.n_particles)))

    @property
    def params_posterior(self):
        return tf.trainable_variables('posterior')

    @property
    def params_prior(self):
        return tf.trainable_variables('prior')

    @property
    def params_likelihood(self):
        return tf.trainable_variables('likelihood')

    def build_kl(self):
        raise NotImplementedError

    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * self.kl_surrogate / self.batch_size

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

        self.infer_latent = self.optimizer.minimize(-self.elbo, var_list=self.params_posterior) \
            if len(self.params_posterior) else tf.no_op()

        self.infer_prior = tf.no_op()
        self.infer_likelihood = self.optimizer.minimize(-self.elbo, var_list=self.params_likelihood)\
            if len(self.params_likelihood) else tf.no_op()

        self.infer_joint = tf.group(self.infer_latent, self.infer_prior, self.infer_likelihood)

    def default_feed_dict(self):
        return {self.coeff_kl: 1., self.coeff_ll: 1.}


class EntropyEstimationFVI(AbstractFVI):
    """
    Function Variational Inference with estimating entropy and computing cross entropy analytically.
    """
    def __init__(self, prior_kernel, posterior, rand_generator, obs_var,
                 input_dim, n_rand, injected_noise, likelihood=None, classification=False,
                 num_classes=None, n_eigen_threshold=0.99, eta=0.):
        super(EntropyEstimationFVI, self).__init__(
            posterior, rand_generator, obs_var, input_dim, n_rand, 
            injected_noise, likelihood=likelihood, classification=classification,
            num_classes=num_classes)
        self.n_eigen_threshold = n_eigen_threshold
        self.eta = eta

        self.prior_kernel = prior_kernel

        self.build_kl()
        self.build_optimizer()

    def build_kl(self):
        # estimate entropy surrogate
        estimator = SpectralScoreEstimator(eta=self.eta, n_eigen_threshold=self.n_eigen_threshold)
        entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_rand)
        # compute analytic cross entropy
        kernel_matrix = self.prior_kernel.K(tf.cast(self.x_rand, tf.float64))
        # if not isinstance(kernel_matrix, AbstractNeuralKernel):
        kernel_matrix += self.injected_noise ** 2 * tf.eye(tf.shape(self.x_rand)[0], dtype=tf.float64)
        mvn_fc = tf.contrib.distributions.MultivariateNormalFullCovariance
        if self.classification:
            mean = tf.zeros([self.n_particles, self.num_classes, tf.shape(self.x_rand)[0]], dtype=tf.float64)
            kernel_matrix = tf.tile(kernel_matrix[None, None, ...], [self.n_particles, self.num_classes, 1, 1])
            prior_dist = mvn_fc(mean, kernel_matrix)
            self.noisy_func_x_rand = tf.transpose(self.noisy_func_x_rand, [0, 2, 1])
        else:
            prior_dist = mvn_fc(tf.zeros([tf.shape(self.x_rand)[0]], dtype=tf.float64), kernel_matrix)
        cross_entropy = -tf.reduce_mean(prior_dist.log_prob(tf.to_double(self.noisy_func_x_rand)))
        self.kl_surrogate = -entropy_sur + tf.to_float(cross_entropy)

    def build_prior_gp(self, gp_model='gpr', gp_likelihood='Gaussian', num_classes=None, init_var=0.1, inducing_points=None,\
                            gp_batch_size=None, num_data=None):
        self.x_gp      = tf.placeholder(tf.float64, shape=[None, self.input_dim], name='x_gp')
        self.x_val_gp  = tf.placeholder(tf.float64, shape=[None, self.input_dim], name='x_val_gp')
        y_type = tf.int32 if self.classification else tf.float64
        self.y_gp      = tf.placeholder(y_type, shape=[None], name='y_gp')
        self.y_val_gp  = tf.placeholder(y_type, shape=[None], name='y_val_gp')

        with tf.variable_scope('prior'):
            if gp_model.lower() == 'gpr':
                self.gp = gfs.models.GPR(self.x_gp, tf.expand_dims(self.y_gp, 1), kern=self.prior_kernel, obs_var=init_var)
            elif gp_model.lower() == 'sgpr':
                self.gp = gfs.models.SGPR(self.x_gp, tf.expand_dims(self.y_gp, 1), kern=self.prior_kernel, Z=inducing_points)
            else:
                if gp_likelihood.lower() == 'multiclass':
                    # Robustmax Multiclass Likelihood
                    invlink = gfs.likelihoods.RobustMax(num_classes)  # Robustmax inverse link function
                    likelihood = gfs.likelihoods.MultiClass(num_classes, invlink=invlink)  # Multiclass likelihood
                else:
                    likelihood = gfs.likelihoods.Gaussian(var=obs_var)
                self.gp = gfs.models.SVGP(self.x_gp, tf.expand_dims(self.y_gp, 1), likelihood=likelihood, \
                                            kern=self.prior_kernel, Z=inducing_points, num_data=num_data, \
                                            num_latent=num_classes, minibatch_size=gp_batch_size)
            self.gp_loss = self.gp.objective

        if gp_likelihood.lower() != 'multiclass':
            self.gp_var = self.gp.likelihood.variance
            self.gp_logstd = tf.log(self.gp.likelihood.variance) * 0.5
        
        self.func_x_pred_gp = self.gp.predict_f_samples(self.x_val_gp, self.n_particles)
        if not self.classification:
            self.func_x_pred_gp = tf.squeeze(self.func_x_pred_gp, -1)

        self.optimizer_gp = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.infer_gp = self.optimizer_gp.minimize(self.gp_loss, var_list=self.params_prior)\
            if len(self.params_prior) else tf.no_op()

        # only optimize kernel params without optimizing GP observation variance
        self.infer_gp_kern = self.optimizer_gp.minimize(
            self.gp_loss, var_list=[v for v in self.params_prior if 'likelihood' not in v.name])

    def build_prior_gp_evaluation(self):
        gp_y_pred, gp_y_pred_var = self.gp.predict_y(self.x_val_gp)
        if self.classification:
            gp_preds = tf.argmax(gp_y_pred, axis=-1, output_type=tf.int32)
            equality = tf.equal(gp_preds, self.y_val_gp)
            self.gp_accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        else:
            self.gp_rmse = tf.sqrt(tf.reduce_mean((gp_y_pred - self.y_val_gp) ** 2))
        self.gp_logll = tf.reduce_mean(self.gp.predict_density(self.x_val_gp, self.y_val_gp))


class KLEstimatorFVI(AbstractFVI):
    """
    Function Variational Inference with estimating the whole KL divergence term.
    """
    def __init__(self, prior_generator, posterior, rand_generator, obs_var,
                 input_dim, n_rand, injected_noise, likelihood=None,
                 n_eigen_threshold=0.99, eta=0.):
        super(KLEstimatorFVI, self).__init__(
            posterior, rand_generator, obs_var,
            input_dim, n_rand, injected_noise, likelihood=likelihood)
        self.prior_gen = prior_generator
        self.n_eigen_threshold = n_eigen_threshold
        self.eta = eta

        self.build_kl()
        self.build_optimizer()

    def build_kl(self):
        # estimate entropy surrogate
        estimator = SpectralScoreEstimator(eta=self.eta, n_eigen_threshold=self.n_eigen_threshold)
        entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_rand)
        
        # estimate cross entropy
        self.prior_func_x_rand = self.prior_gen(self.x_rand, self.n_particles)
        self.noisy_prior_func_x_rand = self.prior_func_x_rand + self.injected_noise * tf.random_normal(
            tf.shape(self.prior_func_x_rand))

        cross_entropy_gradients = estimator.compute_gradients(self.noisy_prior_func_x_rand,
                                                              self.noisy_func_x_rand)
        cross_entropy_sur = -tf.reduce_mean(tf.reduce_sum(
            tf.stop_gradient(cross_entropy_gradients) * self.noisy_func_x_rand, -1))

        self.kl_surrogate = -entropy_sur + cross_entropy_sur
