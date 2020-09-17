import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import tensorflow as tf


def conv2d_with_samples(data, filter, padding='SAME', strides=[1, 1, 1, 1]):
    return tf.map_fn(lambda u:
                tf.nn.conv2d(
                        u[0], u[1],
                        padding=padding,
                        strides=strides),
                        elems=[data, filter],
                        dtype=tf.float32)

def max_pool2d_with_samples(data, ksize, strides, padding):
    return tf.map_fn(lambda u:
                tf.nn.max_pool2d(u,
                        ksize=ksize,
                        padding=padding,
                        strides=strides),
                        elems=data,
                        dtype=tf.float32)

def flatten_rightmost(x, ndims=3):
    leftmost_ndims = len(x.shape.as_list()[:-ndims])
    leftmost_dims = tf.shape(x)[:leftmost_ndims]
    flattened_dim = tf.reduce_prod(x.shape[-ndims:], keepdims=True)
    new_dims = tf.concat([leftmost_dims, flattened_dim], axis=0)
    flattened_x = tf.reshape(x, new_dims)
    return flattened_x
    # """Flatten rightmost dims."""
    # leftmost_ndims = len(x.shape.as_list()[:-ndims]) #tf.rank(x) - ndims
    # leftmost = x.shape[:leftmost_ndims]
    # new_shape = tf.pad(
    #         leftmost,
    #         paddings=[[0, 1]],
    #         constant_values=-1)
    # y = tf.reshape(x, new_shape)
    # if x.shape.ndims is not None:
    #     d = x.shape[leftmost_ndims:]
    #     d = np.prod(d) if d.is_fully_defined() else None
    #     y.set_shape(x.shape[:leftmost_ndims].concatenate(d))
    # return y

def one_hot(data, depth=None, squeeze=True):
    data = np.array(data, dtype=np.int32)
    if not depth:
        depth = data.max() + 1
    data_hot = np.eye(depth)[data]
    if squeeze:
        data_hot = data_hot.squeeze()
    return data_hot

def default_plotting_new():
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.0 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.0 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.0 * plt.rcParams['font.size']
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.xmargin'] = 0


def default_plotting():
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.xmargin'] = 0


def merge_dicts(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


def get_kemans_init(x, k_centers):
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]

    kmeans = MiniBatchKMeans(n_clusters=k_centers,
                             batch_size=k_centers*10).fit(x)
    return kmeans.cluster_centers_


def median_distance_global(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.sqrt(np.sum((x_col - x_row) ** 2, -1))  # [n, n]
    return np.median(dis_a)


def median_distance_local(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row)  # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0) * (x.shape[1] ** 0.5)
