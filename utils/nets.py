import tensorflow as tf
import zhusuan as zs

from .utils import conv2d_with_samples, max_pool2d_with_samples, flatten_rightmost

def get_posterior(name, classification=False):
    if name == 'bnn' or name == 'bnn_relu':
        return bnn_outer(tf.nn.relu, 'bnn', classification)
    if name == 'bnn_tanh':
        return bnn_outer(tf.nn.tanh, 'bnn', classification)
    if name == 'cbnn':
        return bnn_outer(tf.nn.relu, 'cbnn', classification)
    raise NameError('Not a supported name for posterior')


def bnn_outer(activation, type='bnn', classification=False):
    def bnn_inner(layer_sizes, logstd_init=-5.):
        @zs.reuse('posterior')
        def bnn(x, n_particles):
            # x: [batch_size, input_dim]
            h = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])

            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):

                w_mean = tf.get_variable('w_mean_'+str(i), shape=[n_in, n_out],
                                         initializer=tf.contrib.layers.xavier_initializer())
                w_logstd = tf.get_variable('w_logstd_'+str(i), shape=[n_in, n_out],
                                           initializer=tf.constant_initializer(logstd_init))
                w_std = tf.exp(w_logstd)
                ws = w_mean + w_std * \
                    tf.random_normal([n_particles, n_in, n_out])

                b_mean = tf.get_variable('b_mean_' + str(i), shape=[1, n_out],
                                         initializer=tf.zeros_initializer())
                b_logstd = tf.get_variable('b_logstd_' + str(i), shape=[1, n_out],
                                           initializer=tf.constant_initializer(logstd_init))
                b_std = tf.exp(b_logstd)
                bs = b_mean + b_std * tf.random_normal([n_particles, 1, n_out])

                h = tf.matmul(h, ws) + bs
                
                if i < len(layer_sizes) - 2:
                    h = activation(h)
            if classification:
                # h = tf.nn.softmax(h, axis=-1)
                pass
            else:
                h = tf.squeeze(h, -1)
                # h: [n_particles, N]
            return h

        @zs.reuse('posterior')
        def cbnn(x, n_particles):
            # x: [batch, in_height, in_width, in_channels]
            h = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1, 1, 1])

            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-2],
                                                  layer_sizes[1:-1])):
                ###### CONV2D ######
                # filter shape = [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [3, 3, n_in, n_out]
                filter_mean = tf.get_variable('filter_mean_'+str(i), shape=filter_shape,
                                         initializer=tf.contrib.layers.xavier_initializer())
                filter_logstd = tf.get_variable('filter_logstd_'+str(i), shape=filter_shape,
                                           initializer=tf.constant_initializer(logstd_init))
                filter_std = tf.exp(filter_logstd)
                filter_s = filter_mean + filter_std * tf.random_normal([n_particles] + filter_shape)

                fb_mean = tf.get_variable('fb_mean'+str(i), shape=[1, n_out],
                                        initializer=tf.zeros_initializer())
                fb_logstd = tf.get_variable('fb_logstd'+str(i), shape=[1, n_out],
                                            initializer=tf.constant_initializer(logstd_init))
                fb_std = tf.exp(fb_logstd)
                fbs = fb_mean + fb_std * tf.random_normal([n_particles, 1, 1, 1, n_out])
                #### convolve
                h = conv2d_with_samples(h, filter_s, padding='SAME', strides=[1, 1, 1, 1]) + fbs
                #### activate
                h = activation(h)
                ###### MAXPOOL2D ######
                h = max_pool2d_with_samples(h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            #### flatten
            h = flatten_rightmost(h, ndims=3)
            ###### DENSE ######
            n_in = int(h.shape[-1])
            n_out = layer_sizes[-1]
            w_mean = tf.get_variable('w_mean', shape=[n_in, n_out],
                                         initializer=tf.contrib.layers.xavier_initializer())
            w_logstd = tf.get_variable('w_logstd', shape=[n_in, n_out],
                                        initializer=tf.constant_initializer(logstd_init))
            w_std = tf.exp(w_logstd)
            ws = w_mean + w_std * tf.random_normal([n_particles, n_in, n_out])

            b_mean = tf.get_variable('b_mean', shape=[1, n_out],
                                        initializer=tf.zeros_initializer())
            b_logstd = tf.get_variable('b_logstd', shape=[1, n_out],
                                        initializer=tf.constant_initializer(logstd_init))
            b_std = tf.exp(b_logstd)
            bs = b_mean + b_std * tf.random_normal([n_particles, 1, n_out])

            h = tf.matmul(h, ws) + bs
            if classification:
                # h = tf.nn.softmax(h, axis=-1)
                pass
            else:
                h = tf.squeeze(h, -1)
                # h: [n_particles, N]
        if type == 'cbnn':
            return cbnn
        else:
            return bnn

    return bnn_inner
