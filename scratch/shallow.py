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
from scipy.sparse import random as random_sparse

import mne
from mne import SourceEstimate
from mne.minimum_norm import read_inverse_operator
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked
from mne.externals.h5io import read_hdf5, write_hdf5

from shallow_fun import (load_subject_objects, gen_evoked_subject,
                         get_data_batch, get_all_vert_positions,
                         get_largest_dip_positions, get_localization_metrics,
                         eval_error_norm, norm_transpose)

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

    W1 = weight_variable([sensor_dim, source_dim // 4], name='W1')
    b1 = bias_variable([source_dim // 4], name='b1')
    h1 = tf.nn.tanh(tf.matmul(x_sensor, W1) + b1)

    W2 = weight_variable([source_dim // 4, source_dim // 2], name='W2')
    b2 = bias_variable([source_dim // 2], name='b2')
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

    W3 = weight_variable([source_dim // 2, source_dim // 2], name='W3')
    b3 = bias_variable([source_dim // 2], name='b3')
    h3 = tf.nn.tanh(tf.matmul(h2, W3) + b3)

    W4 = weight_variable([source_dim // 2, source_dim], name='W4')
    b4 = bias_variable([source_dim], name='b4')
    yhat = tf.nn.tanh(tf.matmul(h3, W4) + b4)

    # Attach histogram summaries to weight functions
    tf.histogram_summary('W1 Hist', W1)
    tf.histogram_summary('W2 Hist', W2)
    tf.histogram_summary('W3 Hist', W3)
    tf.histogram_summary('W4 Hist', W4)

    h_list = [h1, h2, h3]

    return yhat, h_list, x_sensor


def bernoulli(act, rho):
    """Helper to calculate sparsity penalty based on KL divergence"""

    return (rho * (tf.log(rho) - tf.log(act + 1e-6)) +
            (1 - rho) * (tf.log(1 - rho) - tf.log(1 - act + 1e-6)))


def sparse_objective(sensor_dim, source_dim, yhat, h_list, sess):

    y_source = tf.placeholder(tf.float32, shape=[None, source_dim], name="y_source")
    rho = tf.placeholder(tf.float32, shape=(), name="rho")
    lam = tf.placeholder(tf.float32, shape=(), name="lam")

    diff = y_source - yhat
    error = tf.reduce_sum(tf.squared_difference(y_source, yhat))

    # Remap activations to [0,1]
    act_list = [0.5 * h_obj + 0.5 for h_obj in h_list]

    kl_bernoulli_list = [bernoulli(act, rho) for act in act_list]

    regularizer = sum([tf.reduce_sum(kl_bernoulli_h)
                       for kl_bernoulli_h in kl_bernoulli_list])

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
    n_training_times_noise = 25000
    n_training_times_sparse = 25000
    n_training_times = n_training_times_noise + n_training_times_sparse

    dt = 0.001
    SNR = np.inf
    batch_size = 1000

    n_iter = int(100000)
    rho = 0.1
    lam = 1.

    save_network = True
    fpath_save = op.join('model_subj_{}_iters.meta'.format(n_iter))
    n_training_times = n_training_times_noise + n_training_times_sparse

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    sensor_dim = len(fwd['info']['ch_names'])
    source_dim = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

    sparse_dens = 1. / source_dim  # Density of non-zero vals in sparse training data
    sparse_dist = np.random.randn

    noise_data = np.random.randn(source_dim, n_training_times_noise)
    sparse_data = random_sparse(source_dim, n_training_times_sparse,
                                sparse_dens, random_state=0,
                                dtype=np.float32, data_rvs=sparse_dist).toarray()
    sim_train_data = np.concatenate((noise_data, sparse_data), axis=1)

    print("Simulating and normalizing training data")
    evoked, stc = gen_evoked_subject(sim_train_data, fwd, cov, evoked_info, dt,
                                     SNR)
    # Normalize training data to lie between -1 and 1
    x_train = norm_transpose(evoked.data)
    y_train = norm_transpose(sim_train_data)

    """
    print("Simulating and normalizing testing data")
    evoked, stc = gen_evoked_subject(sim_test_data, fwd, cov, evoked_info, dt,
                                     SNR)
    # Normalize testing data to lie between -1 and 1
    x_test = norm_transpose(evoked.data)
    y_test = norm_transpose(sim_vert_data)

    val_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        x_test, y_test, every_n_steps=20)
    """

    print("Building neural net")
    sess = tf.Session()

    yhat, h_list, x_sensor = make_tanh_network(sensor_dim, source_dim)
    sparse_cost, y_source, tf_rho, tf_lam = sparse_objective(sensor_dim,
                                                             source_dim, yhat,
                                                             h_list, sess)
    merged_summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./train_summaries', sess.graph)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(sparse_cost)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    print("\nSim params\n----------\nn_iter: {}\nn_training_times: {}\nSNR: \
          {}\nbatch_size: {}\n".format(n_iter, n_training_times,
                                       str(SNR), batch_size))
    print("Optimizing...")
    for ii in xrange(n_iter):
        # Get random batch of data
        x_sens_batch, y_src_batch = get_data_batch(x_train, y_train, ii,
                                                   batch_size)

        # Take training step
        feed_dict = {x_sensor: x_sens_batch, y_source: y_src_batch,
                     tf_rho: rho, tf_lam: lam}

        # Save summaries for tensorboard every 10 steps
        if ii % 10 == 0:
            _, obj, summary = sess.run([train_step, sparse_cost,
                                        merged_summaries], feed_dict)
            train_writer.add_summary(summary, ii)
            print("\titer: %d, cost: %.2f" % (ii, obj))
        else:
            _, obj = sess.run([train_step, sparse_cost], feed_dict)

    if save_network:
        saver.save(sess, 'model_{}'.format(struct))
