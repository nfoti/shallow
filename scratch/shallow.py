""" shallow.py

    Run deep neural net on sensor space signals from known source space
    signals.

    Usage:
      shallow.py <megdir> <structdir> [--subj=<subj>]
      shallow.py (-h | --help)

    Options:
      --subj=<subj>     Specify subject to process with structural name
      -h, --help        Show this screen
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as op

import tensorflow as tf
import numpy as np

import mne
from mne import SourceEstimate
from mne.minimum_norm import read_inverse_operator
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked
from mne.externals.h5io import read_hdf5, write_hdf5

from shallow_fun import (load_subject_objects, gen_evoked_subject,
                         get_data_batch, get_all_vert_positions,
                         get_largest_dip_positions, get_localization_metrics,
                         eval_error_norm)

# Entries in structurals and subjects must correspoond,
# i.e. structurals[i] === subjects[i].
structurals = ['AKCLEE_103', 'AKCLEE_104', 'AKCLEE_105', 'AKCLEE_106',
               'AKCLEE_107', 'AKCLEE_109', 'AKCLEE_110', 'AKCLEE_115',
               'AKCLEE_117', 'AKCLEE_118', 'AKCLEE_119', 'AKCLEE_121',
               'AKCLEE_125', 'AKCLEE_126', 'AKCLEE_131', 'AKCLEE_132']
subjects = ['eric_sps_03', 'eric_sps_04', 'eric_sps_05', 'eric_sps_06',
            'eric_sps_07', 'eric_sps_09', 'eric_sps_10', 'eric_sps_15',
            'eric_sps_17', 'eric_sps_18', 'eric_sps_19', 'eric_sps_21',
            'eric_sps_25', 'eric_sps_26', 'eric_sps_31', 'eric_sps_32']

# Removing eric_sps_32/AKCL_132 b/c of vertex issue
structurals = structurals[:-1]
subjects = subjects[:-1]


def weight_variable(shape, name=None):
    init = tf.truncated_normal(shape, stddev=0.11)
    if name is not None:
        return tf.Variable(init, name=name)

    return tf.Variable(init)


def bias_variable(shape, name=None):
    init = tf.constant(0.1, shape=shape)

    if name is not None:
        return tf.Variable(init, name=name)

    return tf.Variable(init)


def make_tanh_network(sensor_dim, source_dim):
    """Function to create neural network"""

    x_sensor = tf.placeholder(tf.float32, shape=[None, sensor_dim],
                              name="x_sensor")

    W1 = weight_variable([sensor_dim, source_dim // 2], name='W1')
    b1 = bias_variable([source_dim // 2], name='b1')
    h1 = tf.nn.tanh(tf.matmul(x_sensor, W1) + b1)

    W2 = weight_variable([source_dim // 2, source_dim // 2], name='W2')
    b2 = bias_variable([source_dim // 2], name='b2')
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

    W3 = weight_variable([source_dim // 2, source_dim], name='W3')
    b3 = bias_variable([source_dim], name='b3')
    yhat = tf.nn.tanh(tf.matmul(h2, W3) + b3)

    # Attach histogram summaries to weight functions
    tf.histogram_summary('W1 Hist', W1)
    tf.histogram_summary('W2 Hist', W2)
    tf.histogram_summary('W3 Hist', W3)

    return yhat, h1, h2, x_sensor


def sparse_objective(sensor_dim, source_dim, yhat, h1, h2, sess):

    y_source = tf.placeholder(tf.float32, shape=[None, source_dim], name="y_source")
    rho = tf.placeholder(tf.float32, shape=(), name="rho")
    lam = tf.placeholder(tf.float32, shape=(), name="lam")

    diff = y_source - yhat
    error = tf.reduce_sum(tf.squared_difference(y_source, yhat))

    # Remap activations to [0,1]
    a1 = 0.5 * h1 + 0.5
    a2 = 0.5 * h2 + 0.5

    kl_bernoulli_h1 = (rho * (tf.log(rho) - tf.log(a1 + 1e-6) + (1 - rho) *
                              (tf.log(1 - rho) - tf.log(1 - a1 + 1e-6))))
    kl_bernoulli_h2 = (rho * (tf.log(rho) - tf.log(a2 + 1e-6) + (1 - rho) *
                              (tf.log(1 - rho) - tf.log(1 - a2 + 1e-6))))
    regularizer = (tf.reduce_sum(kl_bernoulli_h1)
                   + tf.reduce_sum(kl_bernoulli_h2))

    cost = error + lam * regularizer

    # Attach summaries
    tf.scalar_summary('error', error)
    tf.scalar_summary('cost function', cost)

    return cost, y_source, rho, lam

if __name__ == "__main__":

    from docopt import docopt
    argv = docopt(__doc__)

    megdir = argv['<megdir>']
    structdir = argv['<structdir>']

    struct = None
    subj = None
    if argv['--subj']:
        struct = argv['--subj']
        subj = subjects[structurals.index(struct)]

    # Set params
    n_data_times = 50000
    dt = 0.001
    noise_snr = np.inf
    batch_size = 1000

    n_iter = int(500000)
    rho = 0.05
    lam = 1.

    save_network = True
    fpath_save = op.join('model_subj_{}_iters.meta'.format(n_iter))

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    n_verts = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    sim_vert_data = np.random.randn(n_verts, n_data_times)

    print("Simulating and normalizing data")
    evoked, stc = gen_evoked_subject(sim_vert_data, fwd, cov, evoked_info, dt,
                                     noise_snr)
    # Normalize data to lie between -1 and 1
    x_sens_all = np.ascontiguousarray(evoked.data.T)
    x_sens_all /= np.max(np.abs(x_sens_all))
    y_src_all = np.ascontiguousarray(sim_vert_data.T)
    y_src_all /= np.max(np.abs(y_src_all))

    print("Building neural net")
    sess = tf.Session()

    sensor_dim = evoked.data.shape[0]
    source_dim = n_verts
    yhat, h1, h2, x_sensor = make_tanh_network(sensor_dim, source_dim)
    sparse_cost, y_source, tf_rho, tf_lam = sparse_objective(sensor_dim,
                                                             source_dim, yhat,
                                                             h1, h2, sess)
    merged_summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./train_summaries', sess.graph)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(sparse_cost)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    print("\nSim params\n----------\nn_iter: {}\nn_data_points: {}\nSNR: \
          {}\nbatch_size: {}\n".format(n_iter, n_data_times, str(noise_snr),
                                       batch_size))
    print("Optimizing...")
    for ii in xrange(n_iter):
        # Get random batch of data
        x_sens_batch, y_src_batch = get_data_batch(x_sens_all, y_src_all, ii,
                                                   batch_size)

        # Take training step
        feed_dict = {x_sensor: x_sens_batch, y_source: y_src_batch,
                     tf_rho: rho, tf_lam: lam}

        if ii % 10 == 0:
            _, obj, summary = sess.run([train_step, sparse_cost,
                                        merged_summaries], feed_dict)
            train_writer.add_summary(summary, ii)
            print("\titer: %d, cost: %.2f" % (ii, obj))
        else:
            _, obj = sess.run([train_step, sparse_cost], feed_dict)

    if save_network:
        saver.save(sess, 'model_{}'.format(struct))
