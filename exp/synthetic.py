import os.path as osp
import os
import sys
import argparse
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import gpflowSlim as gfs
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from data import SyntheticData
from utils.utils import default_plotting_new as init_plotting
from utils.nets import get_posterior
from core.fvi import EntropyEstimationFVI
from utils.logging import get_logger
from utils.neural_kernel import NeuralSpectralKernel, NeuralGibbsKernel
from utils.utils import median_distance_local

# matplotlib.use('Agg')
# float_type = gfs.settings.tf_float


parser = argparse.ArgumentParser('Synthetic')
parser.add_argument('-d', '--dataset', type=str, default='Dlso')
parser.add_argument('-in', '--injected_noise', type=float, default=0.001)
parser.add_argument('-il', '--init_logstd', type=float, default=-5.)
parser.add_argument('-na', '--n_rand', type=int, default=20)
parser.add_argument('-nh', '--n_hidden', type=int, default=3)
parser.add_argument('-nu', '--n_units', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=3001)
parser.add_argument('-gpe', '--gp_epochs', type=int, default=1000)
parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
parser.add_argument('--train_samples', type=int, default=100)

parser.add_argument('--test_samples', type=int, default=100)
parser.add_argument('--print_interval', type=int, default=200)
parser.add_argument('--test_interval', type=int, default=500)

args = parser.parse_args()
logger = get_logger(args.dataset, 'results/%s/' % args.dataset, __file__)
print = logger.info


############################## load and normalize data ##############################
# dataset = dict(x3=x3_gap_toy, sin=sin_toy)[args.dataset]()
# original_x_train, original_y_train = dataset.train_samples(n_data=50)
dataset = SyntheticData(args.dataset, return_param_values=False, train_test_split=0.8)
original_x_train, original_y_train = dataset.train_samples()
mean_x, std_x = np.mean(original_x_train), np.std(original_x_train)
mean_y, std_y = np.mean(original_y_train), np.std(original_y_train)
train_x = (original_x_train - mean_x) / std_x
train_y = (original_y_train - mean_y) / std_y
original_x_test, original_y_test = dataset.test_samples()
test_x = (original_x_test - mean_x) / std_x
test_y = (original_y_test - mean_y) / std_y
all_x = (dataset.X - mean_x) / std_x
all_y = (dataset.y - mean_y) / std_y

# y_logstd = np.log(dataset.y_std / std_y)
y_logstd = np.log(3. / std_y)

lower_ap = np.minimum(np.min(train_x), np.min(test_x))
upper_ap = np.maximum(np.max(train_x), np.max(test_x))

############################## setup FBNN model ##############################
with tf.variable_scope('prior'):
    # prior_kernel = gfs.kernels.RBF(input_dim=1, name='rbf') + gfs.kernels.Linear(input_dim=1, name='lin')
    # prior_kernel = NeuralSpectralKernel(input_dim=1, name='NSK', Q=1, hidden_sizes=(32, 32))
    # prior_kernel = NeuralGibbsKernel(input_dim=1, name='NGK', hidden_sizes=(5, 5))
    # prior_kernel = gfs.kernels.Periodic(input_dim=1, name='per') + gfs.kernels.RBF(input_dim=1, name='rbf')
    ls = median_distance_local(train_x).astype('float32')
    ls[abs(ls) < 1e-6] = 1.
    prior_kernel = gfs.kernels.RBF(input_dim=1, name='rbf', lengthscales=ls, ARD=False)

with tf.variable_scope('likelihood'):
    obs_log1p = tf.get_variable('obs_log1p', shape=[],
                                initializer=tf.constant_initializer(np.log(np.exp(0.5) - 1.)))
    obs_var = tf.nn.softplus(obs_log1p)**2.


def rand_generator(*arg):
    return tf.random_uniform(shape=[args.n_rand, 1], minval=lower_ap, maxval=upper_ap)

layer_sizes = [1] + [args.n_units] * args.n_hidden + [1]

model = EntropyEstimationFVI(
    prior_kernel, get_posterior('bnn')(layer_sizes, logstd_init=-2.), rand_generator=rand_generator,
    obs_var=obs_var, input_dim=1, n_rand=args.n_rand, injected_noise=args.injected_noise)

# Build the GP prior
model.build_prior_gp(init_var=0.1)

update_op = tf.group(model.infer_latent, model.infer_likelihood)
with tf.control_dependencies([update_op]):
    train_op = tf.assign(obs_log1p, tf.maximum(tf.maximum(
        tf.to_float(tf.log(tf.exp(model.gp_var**0.5) - 1.)), obs_log1p), tf.log(tf.exp(0.05) - 1.)))


############################## training #######################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train the GP
gp_epochs = args.gp_epochs
for epoch in range(gp_epochs):
    feed_dict = {model.x_gp: train_x, model.y_gp: train_y,
                 model.learning_rate_ph: 0.003}
    _, loss, gp_var = sess.run([model.infer_gp_kern, model.gp_loss, model.gp_var],
                       feed_dict=feed_dict)
    if epoch % args.print_interval == 0:
        print(
            '>>> Pretrain GP Epoch {:5d}/{:5d}: Loss={:.5f}'.format(epoch, gp_epochs, loss))

# Plot the GP
test_points = all_x  # or test_x
test_points_vis = dataset.X.squeeze()  # or original_x_test
gp_pred_mu, gp_pred_cov = model.gp.predict_y(test_points) ## to change for test points
mu, cov = sess.run([gp_pred_mu, gp_pred_cov], feed_dict={model.x_gp: train_x, model.y_gp: train_y})
mu, cov = mu.squeeze(), cov.squeeze()
mu, cov = mu * std_y + mean_y, cov * (std_y ** 2)

plt.clf()
figure = plt.figure(figsize=(16, 11), facecolor='white')
init_plotting()

plt.plot(test_points_vis, dataset.y, 'black', label="True function")
plt.plot(test_points_vis, mu, 'orange', label='Mean function')  ## to change for test points
plt.fill_between(test_points_vis, mu - cov ** 0.5, mu + cov ** 0.5, alpha=0.6, color='lightblue') ## to change for test points
plt.scatter(original_x_train, original_y_train, c='green', zorder=10, label='Observations', s=15)
plt.scatter(original_x_test, original_y_test, c='red', zorder=10, label='Test points', s=15)
plt.grid(True)
plt.tick_params(axis='both', bottom='off', top='off', left='off', right='off',
                labelbottom='off', labeltop='off', labelleft='off', labelright='off')
plt.tight_layout()
plt.ylim([np.min(all_y), np.max(all_y)])
plt.tight_layout()
plt.legend()

plt.savefig('results/{}/GP_PRIOR_{}.png'.format(args.dataset, prior_kernel.name))

# Plot GP prior parameters
plt.clf()
figure = plt.figure(figsize=(12, 7), facecolor='white')
init_plotting()
if model.gp.kern.name == 'NSK':
    for q in range(model.gp.kern.Q):  # TODO for Q > 1
        lenf, freqf, varf = model.gp.kern.lengthscale(test_points, q), \
                        model.gp.kern.frequency(test_points, q), \
                        model.gp.kern.variance(test_points, q)
        leng, freq, var = sess.run([lenf, freqf, varf])
        plt.plot(test_points_vis, leng, 'blue', label="Lengthscale")
        plt.plot(test_points_vis, freq, 'orange', label="Frequency")
        plt.plot(test_points_vis, var, 'green', label="Variance")
elif model.gp.kern.name == 'NGK':
    lenf = model.gp.kern.lengthscale(test_points)
    varf = model.gp.kern.variance
    leng, var = sess.run([lenf, varf])
    plt.plot(test_points_vis, leng, 'blue', label="Lengthscale")
    plt.plot(test_points_vis, var*np.ones_like(test_points_vis), 'green', label="Variance")
plt.tight_layout()
plt.legend()
plt.savefig('results/{}/GP_PRIOR_{}_PARAMS.png'.format(args.dataset, prior_kernel.name))

# exit()

# Train the fBNN
test_points = all_x  # or test_x
test_points_vis = dataset.X.squeeze()  # or original_x_test
for epoch in range(args.epochs):
    indices = np.random.permutation(train_x.shape[0])
    train_x, train_y = train_x[indices], train_y[indices]
    feed_dict = {model.x: train_x, model.y: train_y, model.learning_rate_ph: args.learning_rate,
                 model.n_particles: args.train_samples}
    feed_dict.update(model.default_feed_dict())

    _, elbo_sur, kl_sur, logll = sess.run(
        [model.infer_latent, model.elbo, model.kl_surrogate, model.log_likelihood],
        feed_dict=feed_dict)
    if epoch % args.print_interval == 0:
        print('>>> Epoch {:5d}/{:5d} | elbo_sur={:.5f} | logLL={:.5f} | kl_sur={:.5f}'.format(
            epoch, args.epochs, elbo_sur, logll, kl_sur))

    if epoch % args.test_interval == 0:
        y_pred = sess.run(model.func_x_pred,
                          feed_dict={model.x_pred: np.reshape(test_points, [-1, 1]),
                                     model.n_particles: args.test_samples})
        y_pred = y_pred * std_y + mean_y
        mean_y_pred, std_y_pred = np.mean(y_pred, 0), np.std(y_pred, 0)
        ####
        print("fBNN RMSE:", np.sqrt(np.mean((mean_y_pred - dataset.y) ** 2)))
        print("GP RMSE:", np.sqrt(np.mean((mu - dataset.y) ** 2)))
        plt.clf()
        plt.plot(test_points_vis, mean_y_pred)
        plt.plot(test_points_vis, mu)
        plt.plot(test_points_vis, dataset.y)
        plt.ylim([np.min(all_y), np.max(all_y)])
        plt.show()
        ####
        
        plt.clf()
        figure = plt.figure(figsize=(10, 7), facecolor='white')
        init_plotting()

        plt.plot(dataset.X, dataset.y,
                 'black', label="True function")
        plt.plot(test_points_vis, mean_y_pred,
                 'orange', label='Mean function')
        for i in range(5):
            plt.fill_between(test_points_vis, mean_y_pred - i * 0.75 * std_y_pred,
                             mean_y_pred - (i + 1) * 0.75 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='lightblue')
            plt.fill_between(test_points_vis, mean_y_pred + i * 0.75 * std_y_pred,
                             mean_y_pred + (i + 1) * 0.75 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='lightblue')
        plt.scatter(original_x_train, original_y_train, c='green', zorder=10, label='Observations', s=15)
        plt.scatter(original_x_test, original_y_test, c='red', zorder=10, label='Test points', s=15)

        plt.grid(True)
        plt.tick_params(axis='both', bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tight_layout()
        plt.ylim([np.min(all_y), np.max(all_y)])
        plt.tight_layout()
        plt.legend()

        plt.savefig('results/{}/{}_plot_epoch{}.pdf'.format(args.dataset, prior_kernel.name, epoch))
