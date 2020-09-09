import tensorflow as tf
import numpy as np

# import gpflow
# from gpflow import Param

import gpflowSlim.kernels as gfsk
import gpflowSlim as gfs
from gpflowSlim import Param

# from gpflow import settings
float_type = gfs.settings.tf_float
from gpflowSlim import settings


def _create_params(input_dim, output_dim, names):
    def initializer():
        limit = np.sqrt(6. / (input_dim + output_dim))
        # return np.random.uniform(-limit, +limit, (input_dim, output_dim))
        return tf.random.uniform((input_dim, output_dim), -limit, +limit)
    return Param(initializer(), prior=gfs.priors.Gaussian(0, 1), name=names[0]), \
           Param(tf.zeros(output_dim), name=names[1])


def robust_kernel(kern, shape_X):
    with tf.device("/cpu:0"):
        # eigvals = tf.self_adjoint_eigvals(kern)
        eigvals, _ = tf.linalg.eigh(kern)
    min_eig = tf.reduce_min(eigvals)
    jitter = settings.numerics.jitter_level

    def abs_min_eig():
        return tf.Print(tf.abs(min_eig), [min_eig], 'kernel had negative eigenvalue ')

    def zero():
        return float_type(0.0)

    jitter += tf.cond(tf.less(min_eig, 0.0), abs_min_eig, zero)
    return kern + jitter * tf.eye(shape_X, dtype=tf.float64)


class AbstractNeuralKernel(gfsk.Kernel):
    def __init__(self, input_dim, active_dims=None, Q=1, ARD=True, hidden_sizes=None, name=''):
        super().__init__(input_dim, active_dims=active_dims, name=name)
        self.Q = Q
        if hidden_sizes is None:
            hidden_sizes = (32, 32)
        self.num_hidden = len(hidden_sizes)
        self.ARD = ARD

    def _create_nn_params(self, prefix, hidden_sizes, final_size):
        for q in range(self.Q):
            input_dim = self.input_dim
            for level, hidden_size in enumerate(hidden_sizes):
                """name_W = '_{prefix}_{q}_W_{level}'.format(prefix=prefix, q=q, level=level)
                name_b = '_{prefix}_{q}_b_{level}'.format(prefix=prefix, q=q, level=level)
                params = _create_params(input_dim, hidden_size)
                setattr(self, name_W, params[0])
                setattr(self, name_b, params[1])"""
                name_W = '_{prefix}_W_{level}'.format(prefix=prefix, level=level)
                name_b = '_{prefix}_b_{level}'.format(prefix=prefix, level=level)
                if not hasattr(self, name_W):
                    params = _create_params(input_dim, hidden_size, names=(name_W[1:], name_b[1:]))
                    setattr(self, name_W, params[0])
                    setattr(self, name_b, params[1])
                    self._parameters = self._parameters + [getattr(self, name_W),\
                                                           getattr(self, name_b)]
                # input dim for next layer
                input_dim = hidden_size
            name_W_f = '_{prefix}_{q}_W_final'.format(prefix=prefix, q=q)
            name_b_f = '_{prefix}_{q}_b_final'.format(prefix=prefix, q=q)
            params = _create_params(input_dim, final_size, names=(name_W_f[1:], name_b_f[1:]))
            setattr(self, name_W_f, params[0])
            setattr(self, name_b_f, params[1])
            self._parameters = self._parameters + [getattr(self, name_W_f),\
                                                   getattr(self, name_b_f)]

        # @gpflow.params_as_tensors
    def _nn_function(self, x, prefix, q=0, dropout=0.8, final_activation=tf.nn.softplus):
        for level in range(self.num_hidden):
            """W = getattr(self, '{prefix}_{q}_W_{level}'.format(prefix=prefix, q=q, level=level))
            b = getattr(self, '_{prefix}_{q}_b_{level}'.format(prefix=prefix, q=q, level=level))"""
            W = getattr(self, '_{prefix}_W_{level}'.format(prefix=prefix, level=level))
            b = getattr(self, '_{prefix}_b_{level}'.format(prefix=prefix, level=level))
            x = tf.nn.selu(tf.nn.xw_plus_b(x, W, b))  # self-normalizing neural network
            # if dropout < 1.0:
            #     x = tf.contrib.nn.alpha_dropout(x, keep_prob=dropout)
        W = getattr(self, '_{prefix}_{q}_W_final'.format(prefix=prefix, q=q))
        b = getattr(self, '_{prefix}_{q}_b_final'.format(prefix=prefix, q=q))
        output = final_activation(tf.nn.xw_plus_b(x, W, b))
        return output


class NeuralSpectralKernel(AbstractNeuralKernel):  # gpflow.kernels.Kernel
    def __init__(self, input_dim, active_dims=None, Q=1, ARD=True, hidden_sizes=None, name='NSK'):
        super().__init__(input_dim, active_dims=active_dims, Q=Q, hidden_sizes=hidden_sizes, name=name)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  ## ??? maybe deeper?
            for v, final_size in zip(['freq', 'len', 'var'], [input_dim, input_dim, 1]):
                self._create_nn_params(v, hidden_sizes, final_size)

    def variance(self, X, q):
        return self._nn_function(X, 'var', q=q)

    def lengthscale(self, X, q):
        return self._nn_function(X, 'len', q=q)
    
    def frequency(self, X, q):
        return self._nn_function(X, 'freq', q=q)

    # @gpflow.params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        kern = 0.0
        for q in range(self.Q):
            # compute latent function values by the neural network
            freq, freq2 = self._nn_function(X, 'freq', q), self._nn_function(X2, 'freq', q)
            lens, lens2 = self._nn_function(X, 'len', q), self._nn_function(X2, 'len', q)
            var, var2 = self._nn_function(X, 'var', q), self._nn_function(X2, 'var', q)

            if not self.ARD:
                ll = tf.matmul(lens, lens2, transpose_b=True)  # l*l'^T
                # l^2*1^T + 1*(l'^2)^T:
                ll2 = tf.square(lens) + tf.transpose(tf.square(lens2))
                D = square_dist(X_data, X2_data)
                E = tf.sqrt(2 * ll / ll2) * tf.exp(-D/ll2)
            else:
                # compute length-scale term
                Xr = tf.expand_dims(X, 1)  # N1 1 D
                X2r = tf.expand_dims(X2, 0)  # 1 N2 D
                l1 = tf.expand_dims(lens, 1)  # N1 1 D
                l2 = tf.expand_dims(lens2, 0)  # 1 N2 D
                L = tf.square(l1) + tf.square(l2)  # N1 N2 D
                ######D = tf.square((Xr - X2r) / L)  # N1 N2 D
                D = tf.square(Xr - X2r) / L  # N1 N2 D
                D = tf.reduce_sum(D, 2)  # N1 N2
                det = tf.sqrt(2 * l1 * l2 / L)  # N1 N2 D
                det = tf.reduce_prod(det, 2)  # N1 N2
                E = det * tf.exp(-D)  # N1 N2

            # compute cosine term
            muX = (tf.reduce_sum(freq * X, 1, keepdims=True)
                   - tf.transpose(tf.reduce_sum(freq2 * X2, 1, keepdims=True)))
            COS = tf.cos(2 * np.pi * muX)

            # compute kernel variance term
            WW = tf.matmul(var, var2, transpose_b=True)  # w*w'^T

            # compute the q'th kernel component
            kern += WW * E * COS
        if X == X2:
            return robust_kernel(kern, tf.shape(X)[0])
        else:
            return kern

    # @gpflow.params_as_tensors
    def Kdiag(self, X):
        kd = settings.jitter
        for q in range(self.Q):
            kd += tf.square(self._nn_function(X, 'var', q))
        return tf.squeeze(kd)

class NeuralGibbsKernel(AbstractNeuralKernel):  # gpflow.kernels.Kernel
    def __init__(self, input_dim, variance=1.0, ARD=True, active_dims=None, hidden_sizes=None, name='NGK'):
        super().__init__(input_dim, active_dims=active_dims, Q=1, hidden_sizes=hidden_sizes, name=name)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  ## ??? maybe deeper?
            self._create_nn_params('len', hidden_sizes, input_dim)
            self._variance = Param(variance, transform=gfs.transforms.positive, name='variance')
            self._parameters = self._parameters + [self._variance]

    @property
    def variance(self):
        return self._variance.value

    def lengthscale(self, X):
        return self._nn_function(X, 'len')

    # @gpflow.params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        # compute latent function values by the neural network
        lens, lens2 = self._nn_function(X, 'len'), self._nn_function(X2, 'len')
        if not self.ARD:
            ll = tf.matmul(lens, lens2, transpose_b=True)  # l*l'^T
            # l^2*1^T + 1*(l'^2)^T:
            ll2 = tf.square(lens) + tf.transpose(tf.square(lens2))
            D = square_dist(X_data, X2_data)
            kern = self.variance * tf.sqrt(2 * ll / ll2) * tf.exp(-D/ll2)
        else:
            # compute length-scale term
            Xr = tf.expand_dims(X, 1)  # N1 1 D
            X2r = tf.expand_dims(X2, 0)  # 1 N2 D
            l1 = tf.expand_dims(lens, 1)  # N1 1 D
            l2 = tf.expand_dims(lens2, 0)  # 1 N2 D
            L = tf.square(l1) + tf.square(l2)  # N1 N2 D
            ######D = tf.square((Xr - X2r) / L)  # N1 N2 D
            D = tf.square(Xr - X2r) / L  # N1 N2 D
            D = tf.reduce_sum(D, 2)  # N1 N2
            det = tf.sqrt(2 * l1 * l2 / L)  # N1 N2 D
            det = tf.reduce_prod(det, 2)  # N1 N2
            kern = self.variance * det * tf.exp(-D)  # N1 N2

        if X == X2:
            return robust_kernel(kern, tf.shape(X)[0])
        else:
            return kern

    # @gpflow.params_as_tensors
    def Kdiag(self, X):
        kd = settings.jitter + self.variance
        return tf.squeeze(kd)
