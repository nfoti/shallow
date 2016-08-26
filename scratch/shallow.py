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

import mne
from mne import SourceEstimate
from mne.minimum_norm import read_inverse_operator
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked
from mne.externals.h5io import read_hdf5, write_hdf5

import tensorflow as tf
import numpy as np


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


def load_subject_objects(megdatadir, subj, struct):

    print("  %s: -- loading meg objects" % subj)

    fname_fwd = op.join(megdatadir, subj, 'forward',
                        '%s-sss-fwd.fif' % subj)
    fwd = mne.read_forward_solution(fname_fwd, force_fixed=True, surf_ori=True)

    fname_inv = op.join(megdatadir, subj, 'inverse',
                        '%s-55-sss-meg-eeg-fixed-inv.fif' % subj)
    inv = read_inverse_operator(fname_inv)

    fname_epochs = op.join(megdatadir, subj, 'epochs',
                           'All_55-sss_%s-epo.fif' % subj)
    #epochs = mne.read_epochs(fname_epochs)
    #evoked = epochs.average()
    #evoked_info = evoked.info
    evoked_info = mne.io.read_info(fname_epochs)
    cov = inv['noise_cov']

    print("  %s: -- finished loading meg objects" % subj)

    return subj, fwd, inv, cov, evoked_info


def gen_evoked_subject(signal, fwd, cov, evoked_info, dt, noise_snr,
                       seed=None):
    """Function to generate evoked and stc from signal array"""

    vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc = SourceEstimate(sim_vert_data, vertices, tmin=0, tstep=dt)

    evoked = simulate_evoked(fwd, stc, evoked_info, cov, noise_snr,
                             random_state=seed)
    evoked.add_eeg_average_proj()

    return evoked, stc


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.11)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def make_tanh_network(sensor_dim, source_dim):

    x_sensor = tf.placeholder(tf.float32, shape=[None, sensor_dim], name="x_sensor")

    W1 = weight_variable([sensor_dim, source_dim//2])
    b1 = bias_variable([source_dim//2])
    h1 = tf.nn.tanh(tf.matmul(x_sensor, W1) + b1)

    W2 = weight_variable([source_dim//2, source_dim])
    b2 = bias_variable([source_dim])
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

    W3 = weight_variable([source_dim, source_dim])
    b3 = bias_variable([source_dim])
    yhat = tf.nn.tanh(tf.matmul(h2, W3) + b3)

    return yhat, h1, h2, x_sensor


def sparse_objective(sensor_dim, source_dim, yhat, h1, h2, sess):

    y_source = tf.placeholder(tf.float32, shape=[None, source_dim], name="y_source")
    rho = tf.placeholder(tf.float32, shape=(), name="rho")
    lam = tf.placeholder(tf.float32, shape=(), name="lam")

    diff = y_source - yhat
    error = tf.reduce_sum(tf.squared_difference(y_source, yhat))

    # Remap activations to [0,1]
    a1 = 0.5*h1 + 0.5
    a2 = 0.5*h2 + 0.5

    kl_bernoulli_h1 = (rho*(tf.log(rho) - tf.log(a1+1e-6)
                      + (1-rho)*(tf.log(1-rho) - tf.log(1-a1+1e-6))))
    kl_bernoulli_h2 = (rho*(tf.log(rho) - tf.log(a2+1e-6)
                      + (1-rho)*(tf.log(1-rho) - tf.log(1-a2+1e-6))))
    regularizer = (tf.reduce_sum(kl_bernoulli_h1)
                   + tf.reduce_sum(kl_bernoulli_h2))

    cost = error + lam*regularizer

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

    # Params
    n_times = 250
    dt = 0.001
    noise_snr = np.inf

    niter = 100
    rho = 0.05
    lam = 1.

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    n_verts = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    sim_vert_data = np.random.randn(n_verts, n_times)

    print("applying forward operator")
    evoked, stc = gen_evoked_subject(sim_vert_data, fwd, cov, evoked_info, dt,
                                     noise_snr)

    print("building neural net")
    sess = tf.Session()

    # Create neural network
    sensor_dim = evoked.data.shape[0]
    source_dim = n_verts
    
    yhat, h1, h2, x_sensor = make_tanh_network(sensor_dim, source_dim)
    sparse_cost, y_source, tf_rho, tf_lam = sparse_objective(sensor_dim, source_dim,
                                                       yhat, h1, h2, sess)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(sparse_cost) 

    sess.run(tf.initialize_all_variables())

    x_sens = np.ascontiguousarray(evoked.data.T)
    x_sens /= np.max(np.abs(x_sens))
    y_src = np.ascontiguousarray(sim_vert_data.T)
    y_src /= np.max(np.abs(y_src))

    print("optimizing...")
    niter = 100
    for i in xrange(niter):
        _, obj = sess.run([train_step, sparse_cost],
                          feed_dict={x_sensor: x_sens, y_source: y_src,
                                     tf_rho: rho, tf_lam: lam}
                         )

        print("  it: %d i, cost: %.2f" % (i+1, obj))

    # Evaluate net
