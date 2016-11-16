"""eval_model.py

    Evaluate deep neural net on sensor space signals from known source space
    signals. Tests localization error and point spread

    Usage:
      eval_model.py <megdir> <structdir> <nn_fname> [--subj=<subj>]
      eval_model.py (-h | --help)

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
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked
from mne.minimum_norm import apply_inverse

import tensorflow as tf
import numpy as np

from shallow_fun import (load_subject_objects, gen_evoked_subject,
                         get_all_vert_positions, get_largest_dip_positions,
                         get_localization_metrics, eval_error_norm,
                         norm_transpose)
from shallow import make_tanh_network, sparse_objective
from config import common_params, eval_params, conf_structurals, conf_subjects

# Removing eric_sps_32/AKCL_132 b/c of vertex issue
structurals = conf_structurals[:-1]
subjects = conf_subjects[:-1]


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

    # Number of verts to avg when determining est position
    n_avg_verts = eval_params['n_avg_verts']
    # Probably should be <= 1000 to avoid mem problems
    n_test_verts = eval_params['n_test_verts']

    sess = tf.Session()

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    vert_list = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    n_verts = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

    print("Simulating and normalizing data")
    sensor_dim = len(fwd['info']['ch_names'])
    source_dim = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

    print("Reconstructing model and restoring saved weights")
    # Reconstruct network
    network_dims = [source_dim // 2, source_dim // 2, source_dim]
    yhat, h_list, x_sensor = make_tanh_network(sensor_dim, source_dim, network_dims)
    sparse_cost, y_source, tf_rho, tf_lam = sparse_objective(sensor_dim, source_dim,
                                                             yhat, h_list,
                                                             sess)
    saver = tf.train.Saver()
    saver.restore(sess, model_fname)

    print("\nEvaluating deep learning approach...\n")

    # Simulate unit dipole activations

    rand_verts = np.sort(np.random.choice(range(n_verts), n_test_verts,
                                          replace=False))
    sim_vert_data = np.eye(n_verts)[:, rand_verts]
    evoked, stc = gen_evoked_subject(sim_vert_data, fwd, cov, evoked_info,
                                     common_params['dt'],
                                     common_params['SNR'])

    # Normalize data and transpose so it's (n_observations x n_chan)
    x_test = norm_transpose(evoked.data)
    y_test = norm_transpose(stc.data)

    # Ground truth dipole positions
    vert_positions = get_all_vert_positions(inv['src'])
    true_act_positions = vert_positions[rand_verts, :]

    feed_dict = {x_sensor: x_test, y_source: y_test,
                 tf_rho: common_params['rho'], tf_lam: common_params['lam']}
    src_est_dl = sess.run(yhat, feed_dict)
    stc_dl = SourceEstimate(src_est_dl.T, vertices=vert_list, subject=struct,
                            tmin=0, tstep=common_params['dt'])

    # Calculate vector norm error
    error_norm_dl = eval_error_norm(y_test, src_est_dl)

    # Get position of most active dipoles and calc accuracy metrics (in meters)
    est_act_positions = get_largest_dip_positions(src_est_dl, n_avg_verts,
                                                  vert_positions)
    accuracy_dl, point_spread_dl = get_localization_metrics(true_act_positions,
                                                            est_act_positions)

    print("\nEvaluating standard linear approach...\n")
    #
    # Evaluate standard MNE methods
    #
    stc_std = apply_inverse(evoked, inv, method=eval_params['linear_inv'])
    src_est_std = stc_std.data.T

    # Calculate vector norm error
    error_norm_std = eval_error_norm(y_test, src_est_std)
    est_act_positions = get_largest_dip_positions(src_est_std, n_avg_verts,
                                                  vert_positions)
    accuracy_std, point_spread_std = get_localization_metrics(true_act_positions,
                                                              est_act_positions)

    sess.close()
    print('\bShallow; error norm average for {} verts: {:0.4f}'.format(
        n_test_verts, np.mean(error_norm_dl)))
    print('Linear method: error norm average for {} verts: {:0.4f}\n'.format(
        n_test_verts, np.mean(error_norm_std)))
    print('Shallow; Loc. accuracy: {:0.5f}, Avg. Point spread: {:0.5f}'.format(
        accuracy_dl, np.mean(point_spread_dl)))
    print('Linear method; Loc. accuracy: {:0.5f}, Avg. Point spread: {:0.5f}\n'.format(
        accuracy_std, np.mean(point_spread_std)))
