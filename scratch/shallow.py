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
from config import (common_params, training_params, conf_structurals,
                    conf_subjects)

# Removing eric_sps_32/AKCL_132 b/c of vertex issue
structurals = conf_structurals[:-1]
subjects = conf_subjects[:-1]
seed = 0


def weight_variable(shape, name=None):
    init = tf.truncated_normal(shape, stddev=0.1)
    if name is not None:
        return tf.Variable(init, name=name)

    return tf.Variable(init)


def bias_variable(shape, name=None):
    init = tf.constant(0.1, shape=shape)

    if name is not None:
        return tf.Variable(init, name=name)

    return tf.Variable(init)


def make_tanh_network(sensor_dim, source_dim, dims):
    """Function to create neural network"""

    x_sensor = tf.placeholder(tf.float32, shape=[None, sensor_dim],
                              name="x_sensor")

    W_list, b_list, h_list = [], [], []
    dims.insert(0, sensor_dim)  # Augment with input layer dim

    # Loop through and create network layer at each step
    for di, (dim1, dim2) in enumerate(zip(dims[:-1], dims[1:])):
        W_list.append(weight_variable([dim1, dim2], name='W%i' % di))
        b_list.append(bias_variable([dim2], name='b%i' % di))

        # Handle input layer separately
        if di == 0:
            h_list.append(tf.nn.tanh(tf.matmul(x_sensor, W_list[-1]) +
                                     b_list[-1]))
        else:
            h_list.append(tf.nn.tanh(tf.matmul(h_list[-1], W_list[-1]) +
                                     b_list[-1]))

        # Attach histogram summaries to weight functions
        tf.histogram_summary('W%i Hist' % di, W_list[-1])

    # Return y_hat (final h_list layer), rest of h_list, and data placeholder
    return h_list[-1], h_list[:-1], x_sensor


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

    n_training_times = training_params['n_training_times_noise'] + \
        training_params['n_training_times_sparse']

    n_training_iters = training_params['n_training_iters']

    save_network = True
    fpath_save = op.join('model_subj_{}_iters.meta'.format(n_training_iters))
    n_training_times = training_params['n_training_times_noise'] + \
        training_params['n_training_times_sparse']

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    sensor_dim = len(fwd['info']['ch_names'])
    source_dim = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    network_dims = [source_dim // 2, source_dim // 2, source_dim]

    sparse_dens = 1. / source_dim  # Density of non-zero vals in sparse training data
    sparse_dist = np.random.randn

    noise_data = np.random.randn(source_dim, training_params['n_training_times_noise'])
    sparse_data = random_sparse(source_dim, training_params['n_training_times_sparse'],
                                sparse_dens, random_state=seed,
                                dtype=np.float32, data_rvs=sparse_dist).toarray()
    sim_train_data = np.concatenate((noise_data, sparse_data), axis=1)

    print("Simulating and normalizing training data")
    evoked, stc = gen_evoked_subject(sim_train_data, fwd, cov, evoked_info, common_params['dt'],
                                     common_params['SNR'])
    # Normalize training data to lie between -1 and 1
    # XXX: Appropriate to do this? Maybe need to normalize src space only
    # before generating sens data
    x_train = norm_transpose(evoked.data)
    y_train = norm_transpose(sim_train_data)

    print("Building neural net")
    sess = tf.Session()
    yhat, h_list, x_sensor = make_tanh_network(sensor_dim, source_dim, network_dims)
    sparse_cost, y_source, tf_rho, tf_lam = sparse_objective(sensor_dim,
                                                             source_dim, yhat,
                                                             h_list, sess)
    merged_summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./train_summaries', sess.graph)

    train_step = tf.train.AdamOptimizer(
        training_params['opt_lr']).minimize(sparse_cost)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    print("\nSim params\n----------\nn_iter: {}\nn_training_times: {}\nSNR: \
          {}\nbatch_size: {}\n".format(n_training_iters, n_training_times,
                                       str(common_params['SNR']), training_params['batch_size']))
    print("Optimizing...")
    for ii in xrange(n_training_iters):
        # Get random batch of data
        x_sens_batch, y_src_batch = get_data_batch(x_train, y_train,
                                                   training_params['batch_size'], seed=ii)

        # Take training step
        feed_dict = {x_sensor: x_sens_batch, y_source: y_src_batch,
                     tf_rho: common_params['rho'],
                     tf_lam: common_params['lam']}

        # Save summaries for tensorboard every 10 steps
        if ii % 10 == 0:
            _, obj, summary = sess.run([train_step, sparse_cost,
                                        merged_summaries], feed_dict)
            train_writer.add_summary(summary, ii)
            print("\titer: %04i, cost: %.2f" % (ii, obj))
        else:
            _, obj = sess.run([train_step, sparse_cost], feed_dict)

    if save_network:
        save_fold = 'saved_models'
        if not os.path.isdir(save_fold):
            os.mkdir(save_fold)

        saver.save(sess, save_fold + '/model_{}'.format(struct))

    sess.close()
