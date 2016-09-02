"""eval_model.py

    Evaluate deep neural net on sensor space signals from known source space
    signals. Tests localization error and point spread

    Usage:
      shallow.py <megdir> <structdir> <nn_fname> [--subj=<subj>]
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
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked

import tensorflow as tf
import numpy as np

from shallow_fun import (load_subject_objects, gen_evoked_subject,
                         get_all_vert_positions, get_largest_dip_positions,
                         get_localization_metrics, eval_error_norm)
from shallow import make_tanh_network, sparse_objective


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


if __name__ == "__main__":

    from docopt import docopt
    argv = docopt(__doc__)

    megdir = argv['<megdir>']
    structdir = argv['<structdir>']
    model_fname = argv['<nn_fname>']

    struct = None
    subj = None
    if argv['--subj']:
        struct = argv['--subj']
        subj = subjects[structurals.index(struct)]

    # Set params
    n_data_times = 1000
    dt = 0.001
    noise_snr = np.inf
    rho = 0.05
    lam = 1.

    sess = tf.Session()

    #saver = tf.train.import_meta_graph('model_AKCLEE_107.meta')
    #saver.restore(sess, model_fname.split('.')[:-1])
    #yhat = tf.get_collection(['yhat'][0])

    print('\nLoaded saved model: %s\n' % model_fname)
    print("\nEvaluating...\n")

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    n_verts = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

    # Simulate identity activations
    sim_vert_data = np.eye(n_verts)[:, :500] * 0.1
    evoked, stc = gen_evoked_subject(sim_vert_data, fwd, cov, evoked_info, dt,
                                     noise_snr)

    print("Simulating and normalizing data")
    sens_dim = evoked.data.shape[0]
    src_dim = n_verts

    # Reconstruct network
    yhat, h1, h2, x_sensor = make_tanh_network(sens_dim, src_dim)
    sparse_cost, y_source, tf_rho, tf_lam = sparse_objective(sens_dim, src_dim,
                                                             yhat, h1, h2,
                                                             sess)
    saver = tf.train.Saver()
    saver.restore(sess, 'model_AKCLEE_107')

    #
    # Evaluate network
    #
    x_sens = np.ascontiguousarray(evoked.data.T)
    y_src = np.ascontiguousarray(stc.data.T)

    feed_dict = {x_sensor: x_sens, y_source: y_src, tf_rho: rho, tf_lam: lam}
    src_est = sess.run(yhat, feed_dict)

    # Calculate vector norm error
    error_norm = eval_error_norm(y_src, src_est)
    dip_positions = get_all_vert_positions(inv['src'])

    # Ground truth pos
    #activation_positions = dip_positions[rand_verts, :]
    activation_positions = dip_positions[:500, :]
    # Get position of most active dipoles
    largest_dip_positions = get_largest_dip_positions(src_est, 25,
                                                      dip_positions)
    # Calculate accuracy metrics (in meters)
    accuracy, point_spread = get_localization_metrics(activation_positions,
                                                      largest_dip_positions)
    print('Accuracy: {}, Avg. Point spread: {}'.format(accuracy,
                                                       np.mean(point_spread)))

    n_eval_data_times = 1000  # Probably should be <= 1000 to avoid mem problems

    print("\nEvaluating...\n")

    # Simulate identity activations
    #rand_verts = np.sort(np.random.randint(0, n_verts, n_eval_data_times))
    #sim_vert_eval_data = np.eye(n_verts)[:, rand_verts] * 0.1
    sim_vert_eval_data = np.eye(n_verts)[:, :500] * 0.1
    evoked_eval, stc_eval = gen_evoked_subject(sim_vert_eval_data, fwd, cov,
                                               evoked_info, dt, noise_snr)

    x_sens_eval = np.ascontiguousarray(evoked_eval.data.T)
    y_src_eval = np.ascontiguousarray(stc_eval.data.T)

    feed_dict = {x_sensor: x_sens_eval, y_source: y_src_eval,
                 tf_rho: rho, tf_lam: lam}
    src_est = sess.run(yhat, feed_dict)

    # Calculate vector norm error
    error_norm = eval_error_norm(y_src_eval, src_est)

    dip_positions = get_all_vert_positions(inv['src'])

    # Ground truth pos
    activation_positions = dip_positions[:500, :]
    # Get position of most active dipoles
    largest_dip_positions = get_largest_dip_positions(src_est, 25,
                                                      dip_positions)
    # Calculate accuracy metrics (in meters)
    accuracy, point_spread = get_localization_metrics(activation_positions,
                                                      largest_dip_positions)
    print('Accuracy: {}, Avg. Point spread: {}'.format(accuracy,
                                                       np.mean(point_spread)))

