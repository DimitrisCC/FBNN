"""
Implements the following spectral kernels:
    Spectral Mixture (SM) by Wilson (2013) [stationary]
"""
import tensorflow as tf
import numpy as np

import gpflowSlim as gfs
from gpflowSlim.kernels import Kernel, Stationary, Sum, Product  # used to derive new kernels
from gpflowSlim import Param
from gpflowSlim import transforms

from gpflowSlim import settings
float_type = settings.dtypes.float_type


def square_dist(X, X2):
    Xs = tf.reduce_sum(tf.square(X), 1)
    X2s = tf.reduce_sum(tf.square(X2), 1)
    return -2 * tf.matmul(X, X2, transpose_b=True) + tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))


class SMKernelComponent(Stationary):
    """
    Spectral Mixture kernel.
    """
    def __init__(self, input_dim, variance=1.0, lengthscales=None, 
                 frequency=1.0, active_dims=None, ARD=False, component_num=0, name='SM-Component'):
        Stationary.__init__(self, input_dim=input_dim, variance=variance, lengthscales=lengthscales,
                            active_dims=active_dims, ARD=ARD, name=name+str(component_num))
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._frequency = Param(frequency, transforms.positive, dtype=float_type, name='freq'+str(component_num))
            self._frequency.prior = gfs.priors.Exponential(1.0)
            self._variance.prior = gfs.priors.LogNormal(0, 1)
        self._parameters = self._parameters + [self._frequency]

    @property
    def frequency(self):
        return self._frequency

    # @gpflow.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X
        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        freq = tf.expand_dims(self.frequency, 0)
        freq = tf.expand_dims(freq, 0)  # 1 x 1 x D
        r = tf.reduce_sum(2.0 * np.pi * freq * (f - f2), 2)
        return self.variance * tf.exp(-2.0*np.pi**2*self.scaled_square_dist(X, X2)) * tf.cos(r)
    
    # @gpflow.params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    # @gpflow.params_as_tensors
    def scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return dist


def SMKernel(input_dim, active_dims=None, Q=1, variances=None, frequencies=None,
             lengthscales=None, max_freq=1.0, max_len=1.0, ARD=False, name='SMK'):
    """
    Initialises a SM kernel with Q components. Optionally uses a given initialisation,
    otherwise uses a random initialisation.
    max_freq: Nyquist frequency of the signal, used to initialize frequencies
    max_len: range of the inputs x, used to initialize length-scales
    """
    if variances is None:
        variances = [1./Q for _ in range(Q)]
    if frequencies is None:
        frequencies = [np.random.rand(input_dim)*max_freq for _ in range(Q)]
    if lengthscales is None:
        lengthscales = [np.abs(max_len*np.random.randn(input_dim if ARD else 1)) for _ in range(Q)]
    kerns = [SMKernelComponent(input_dim, active_dims=active_dims, variance=variances[i],
                               frequency=frequencies[i], lengthscales=lengthscales[i], ARD=ARD, component_num=i)
             for i in range(Q)]
    sm_kernel = Sum(kerns)
    sm_kernel.name = name
    return sm_kernel
