from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import time

import numpy as np
import argparse
from scipy import stats
import gpflowSlim as gfs
from gpflowSlim.neural_kernel_network import NKNWrapper, NeuralKernelNetwork
import tensorflow as tf
import shutil

from data import uci_woval
from utils.kernels import KernelWrapper
from utils.logging import get_logger
from utils.nets import get_posterior
from core.fvi import EntropyEstimationFVI
from utils.neural_kernel import NeuralSpectralKernel, NeuralGibbsKernel


parser = argparse.ArgumentParser('Classification')
parser.add_argument('-d', '--dataset', type=str, default='iris')
#tuning
parser.add_argument('-na', '--n_rand', type=int, default=5)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-aiter', '--anneal_lr_iters', type=int, default=100000000)
parser.add_argument('-arate', '--anneal_lr_ratio', type=int, default=0.1)
parser.add_argument('-kernel', '--kernel', default='rbf')
parser.add_argument('-kh', '--kernel_hidden', type=int, default=32)
parser.add_argument('-Q', '--components', type=int, default=3)

parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

str_ = 'regression/{}/NA{}_LR{}_AI{}/{}_'.format(
    args.dataset, args.n_rand, args.learning_rate, args.anneal_lr_iters, args.seed)
logger = get_logger(args.dataset, 'results/'+str_, __file__)
print = logger.info

############################## Prior Kernel  ##############################
def NKNInfo(input_dim):
    kernel = [
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True, 'name': 'Linear1'}},
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True, 'name': 'Linear2'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'ARD': True, 'name': 'RBF1'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'ARD': True, 'name': 'RBF2'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'ARD': True, 'alpha': 0.1, 'name': 'RatQuad1'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'ARD': True, 'alpha': 1., 'name': 'RatQuad2'}}
    ]
    wrapper = [
            {'name': 'Linear',  'params': {'input_dim': 6, 'output_dim': 8, 'name': 'layer1'}},
            {'name': 'Product', 'params': {'input_dim': 8, 'step': 2,       'name': 'layer2'}},
            {'name': 'Linear',  'params': {'input_dim': 4, 'output_dim': 4, 'name': 'layer3'}},
            {'name': 'Product', 'params': {'input_dim': 4, 'step': 2,       'name': 'layer4'}},
            {'name': 'Linear',  'params': {'input_dim': 2, 'output_dim': 1, 'name': 'layer5'}}
    ]
    return kernel, wrapper

############################## Likelihood ##############################
class MulticlassLikelihood:
    def __init__(self, classes):
        with tf.variable_scope('likelihood'):
            self.lkh = tf.nn.sparse_softmax_cross_entropy_with_logits

    def forward(self, h, y=None):
        if y is None:
            return tf.distributions.Categorical(h).sample()
        else:
            return self.lkh(h)

def run():
    data = uci_woval(args.dataset, args.seed*666)
    tf.set_random_seed(args.seed * 666)
    np.random.seed(args.seed * 666)

    ############################## load and normalize data ##############################
    x_test, y_test = data.x_test, data.y_test
    permutation = np.random.permutation(data.x_train.shape[0])
    n_test = x_test.shape[0]
    x_train, y_train = data.x_train[permutation[n_test:]], data.y_train[permutation[n_test:]]
    x_valid, y_valid = data.x_train[permutation[:n_test]], data.y_train[permutation[:n_test]]
    y_train_onehot = tf.one_hot(y_train, tf.reduce_max(y_train))
    y_valid_onehot = tf.one_hot(y_valid, tf.reduce_max(y_valid))
    print('TRAIN : {}'.format(x_train.shape))
    print('VALID : {}'.format(x_valid.shape))
    print('TEST  : {}'.format(x_test.shape))
    N, D = x_train.shape
    lower_ap, upper_ap = np.min(x_train, 0), np.max(x_train, 0)
    lower_ap = lower_ap - 0.5 * (upper_ap - lower_ap)
    upper_ap = upper_ap + 0.5 * (upper_ap - lower_ap)

    ############################## setup FBNN model ##############################
    with tf.variable_scope('prior'):
        if args.kernel.upper() == 'NKN':
            kernel, wrapper = NKNInfo(input_dim=D)
            wrapper = NKNWrapper(wrapper)
            kern = NeuralKernelNetwork(D, KernelWrapper(kernel), wrapper)
        elif args.kernel.upper() == 'NSK' or ('NEURALSPECTRAL' in args.kernel.upper()):
            kern = NeuralSpectralKernel(input_dim=D, name='NSK', Q=args.components, hidden_sizes=(args.kernel_hidden, args.kernel_hidden))
        elif args.kernel.upper() == 'GIBBS' or args.kernel.upper() == 'NGK':
            kern = NeuralGibbsKernel(input_dim=D, name='NGK', hidden_sizes=(args.kernel_hidden, args.kernel_hidden))
        elif args.kernel.upper() == 'RBF':
            kern = gfs.kernels.RBF(input_dim=D, name='rbf', ARD=True)

    likelihood = MulticlassLikelihood(classes=y_train.shape[1])

    def rand_generator(*arg):
        return tf.random_uniform(shape=[args.n_rand, D], minval=lower_ap, maxval=upper_ap)

    layer_sizes = [D] + [100] + [1]
    model = EntropyEstimationFVI(
        kern, get_posterior('bnn')(layer_sizes, logstd_init=-5.), rand_generator=rand_generator,
        obs_var=None, input_dim=D, n_rand=args.n_rand, injected_noise=0.01,
        likelihood=likelihood)
    model.build_prior_gp(init_var=0.1)

    ################### LIKELIHOOD ###################
    ##### belief-match softmax to categorical with dirichlet prior
    ##### using ELBO
    y_onehot = tf.one_hot(model.y, tf.reduce_max(model.y))
    y_obs = tf.tile(tf.expand_dims(y_onehot, 0), [model.n_particles, 1])
    coeff = 1e-2
    # dirichlet prior
    alpha_prior = tf.ones_like(model.func_x) * 1.0
    # concentration parameters
    alpha = tf.exp(model.func_x)
    alpha0 = tf.reduce_sum(alpha, -1)
    alpha_prior0 = tf.reduce_sum(alpha_prior, -1)
    alpha_ans = tf.reduce_sum(alpha * y_obs, axis=-1)

    # expectation of log-likelihood
    ll_loss = tf.digamma(alpha_ans) - tf.digamma(alpha0)

    # KL(Dirichlet(preds) || Dirichlet(prior))
    kl_loss1 = tf.lgamma(alpha0) - tf.reduce_sum(tf.lgamma(alpha), -1) - tf.lgamma(alpha_prior0)
    kl_loss2 = tf.reduce_sum((alpha - alpha_prior) \
                * (tf.digamma(alpha) - tf.digamma(tf.expand_dims(alpha0, -1))), -1)
    kl_loss = kl_loss1 + kl_loss2
    # # or simply:
    # dirD = tf.distributions.Dirichlet
    # kl_loss = tf.distributions.kl_divergence(dirD(alpha), dirD(alpha_prior))

    elbo_loss = ll_loss + kl_loss / N # or *coeff
    infer_likelihood = model.optimizer.minimize(elbo_loss, var_list=model.params_likelihood)

    ####### Initialize session #######
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=10,
                                            inter_op_parallelism_threads=10))
    sess.run(tf.global_variables_initializer())

    ############################## train GP firstly ##############################
    gp_batch_size = 1000
    epochs = 2 #10000 * gp_batch_size // N
    flag = False
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        iter = N // gp_batch_size
        if flag:
            break
        for idx in range(iter):
            sub_indices = indices[idx * gp_batch_size : (idx+1) * gp_batch_size]
            x_batch, y_batch = x_train[sub_indices, :], y_train[sub_indices]
            _, loss, var = sess.run(
                [model.infer_gp, model.gp_loss, model.gp_var],
                feed_dict={model.x_gp: x_batch, model.y_gp: y_batch, model.learning_rate_ph: args.learning_rate})
            if idx % 100 == 0:
                print('Epoch %d/%d -- Iter %d/%d -- GP Loss = %.4f -- Obs Var = %.4f' % (epoch, epochs, idx, iter, loss, var))
            if var < 2e-5:
                flag = True
                break

     ############################## evaluation function ##############################
    def eval(test_input, test_output, for_gp=False):
        test_output = test_output.squeeze()
        rmse, lld, ns = [], [], []
        for id in range(test_input.shape[0] // 1000 + 1):
            x_batch = test_input[id * 1000 : (id+1) * 1000]
            y_batch = test_output[id * 1000 : (id+1) * 1000]
            # # if for_gp:
            # #     r, l = sess.run([
            # #         model.gp_rmse, model.gp_logll],
            # #         feed_dict={model.x_gp: x_train, model.y_gp: y_train,
            # #             model.x_val_gp: x_batch, model.y_val_gp: y_batch})
            # else:
            r, l = sess.run(
                [model.eval_rmse, model.eval_lld],
                feed_dict={model.x: x_batch, model.y: y_batch, model.n_particles: 100}
            )
            rmse.append(r)
            lld.append(l)
            ns.append(x_batch.shape[0])

        rmse = np.sqrt(np.sum([r**2. * n for r, n in zip(rmse, ns)]) / np.sum(ns))
        log_likelihood = np.sum([l * n for l, n in zip(lld, ns)]) / np.sum(ns)
        return rmse * std_y_train, log_likelihood - np.log(std_y_train)
    
    ############################## test the GP ################################
    model.build_prior_gp_evaluation()
    gp_rmse, gp_logll = eval(x_test, y_test, for_gp=True)
    print('>>> GP Prior with {} fit: rmse={:.5f} | lld={:.5f}'.format(kern.name, gp_rmse.item(), gp_logll.item()))

    ############################## train FBNN ##############################
    batch_size = 500
    best_valid_rmse, best_valid_likelihood = np.float('inf'), -np.float('inf')
    best_test_rmse, best_test_likelihood = np.float('inf'), -np.float('inf')
    epochs = 2 #80000 * batch_size // N
    lr = args.learning_rate
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        iter = N // batch_size
        for idx in range(iter):
            if epoch * iter + idx == args.anneal_lr_iters - 1:
                print('---------- Anneal Learning Rate by %.5f' % args.anneal_lr_ratio)
                lr = args.learning_rate * args.anneal_lr_ratio

            sub_indices = indices[idx * batch_size : (idx+1) * batch_size]
            x_batch, y_batch = x_train[sub_indices, :], y_train[sub_indices]
            fd = {model.x: x_batch, model.y: y_batch, model.n_particles: 100,
                  model.learning_rate_ph: lr}
            fd.update(model.default_feed_dict())
            _, _, elbo, kl, eq = sess.run(
                [model.infer_latent, infer_likelihood, model.elbo, model.kl_surrogate, model.log_likelihood],
                feed_dict=fd)
            if idx % 20 == 0:
                print('Epoch %d/%d -- Iter %d/%d -- ELBO = %.4f -- KL = %.4f -- Log Likelihood = %.4f' % (
                    epoch, epochs, idx, iter, elbo, kl/N, eq))

        valid_rmse, valid_likelihood = eval(x_valid, y_valid)
        if valid_likelihood > best_valid_likelihood:
            print('*************** New best valid resbanditult **********************')
            best_valid_rmse, best_valid_likelihood = valid_rmse, valid_likelihood
            test_rmse, test_likelihood = eval(x_test, y_test)
            best_test_rmse, best_test_likelihood = test_rmse, test_likelihood
      
        logger.info('Epoch %d/%d -- NOW: valid rmse = %.5f -- valid ll = %.5f' % (
            epoch, epochs, valid_rmse, valid_likelihood))
        logger.info('Epoch %d/%d -- Till NOW: best valid rmse = %.5f -- best valid ll = %.5f' % (
            epoch, epochs, best_valid_rmse, best_valid_likelihood))
        logger.info('Epoch %d/%d -- Till NOW: best test rmse = %.5f -- best test ll = %.5f' % (
            epoch, epochs, best_test_rmse, best_test_likelihood))
        logger.info('>>> GP Prior with {} fit: rmse={:.5f} | lld={:.5f}'.format(kern.name, gp_rmse.item(), gp_logll.item()))


if __name__ == "__main__":
    begin_time = time.time()
    run()
    logger.info('Time Used = {} s'.format(time.time() - begin_time))
