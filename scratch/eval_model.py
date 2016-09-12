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
from mne import SourceEstimate
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked
from mne.minimum_norm import apply_inverse

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
    dt = 0.001
    noise_snr = np.inf
    rho = 0.05
    lam = 1.
    n_avg_verts = 25  # Number of verts to avg when determining est position
    n_eval_data_times = 1000  # Probably should be <= 1000 to avoid mem problems

    sess = tf.Session()

    #saver = tf.train.import_meta_graph('model_AKCLEE_107.meta')
    #saver.restore(sess, model_fname.split('.')[:-1])
    #yhat = tf.get_collection(['yhat'][0])

    print('\nLoaded saved model: %s\n' % model_fname)
    print("\nEvaluating...\n")

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    vert_list = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
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
    # Evaluate deep learning network
    #

    print("\nEvaluating deep learning approach...\n")

    # Simulate unit dipole activations
    #rand_verts = np.sort(np.random.randint(0, n_verts, n_eval_data_times))
    #sim_vert_eval_data = np.eye(n_verts)[:, rand_verts] * 0.1
    sim_vert_data = np.eye(n_verts)[:, :n_eval_data_times]
    evoked, stc = gen_evoked_subject(sim_vert_data, fwd, cov, evoked_info, dt,
                                     noise_snr)

    x_sens = np.ascontiguousarray(evoked.data.T)
    y_src = np.ascontiguousarray(stc.data.T)
    #TODO: normalize data? Transfer normalization multipliers? (maybe with sklearn)

    # Ground truth dipole positions
    vert_positions = get_all_vert_positions(inv['src'])
    true_act_positions = vert_positions[:n_eval_data_times, :]

    feed_dict = {x_sensor: x_sens, y_source: y_src, tf_rho: rho, tf_lam: lam}
    src_est_dl = sess.run(yhat, feed_dict)
    stc_dl = SourceEstimate(src_est_dl.T, vertices=vert_list, subject=struct,
                            tmin=0, tstep=0.001)

    # Calculate vector norm error
    error_norm_dl = eval_error_norm(y_src, src_est_dl)

    # Get position of most active dipoles and calc accuracy metrics (in meters)
    est_act_positions = get_largest_dip_positions(src_est_dl, n_avg_verts,
                                                  vert_positions)
    accuracy_dl, point_spread_dl = get_localization_metrics(true_act_positions,
                                                            est_act_positions)

    print("\nEvaluating deep learning approach...\n")
    #
    # Evaluate standard MNE methods
    #
    stc_mne = apply_inverse(evoked, inv, method='sLORETA')
    src_est_mne = stc_mne.data.T

    # Calculate vector norm error
    error_norm_mne = eval_error_norm(y_src, src_est_mne)
    est_act_positions = get_largest_dip_positions(src_est_mne, n_avg_verts,
                                                  vert_positions)
    accuracy_mne, point_spread_mne = get_localization_metrics(true_act_positions,
                                                              est_act_positions)

    print('\bDeep learning; error norm average for {} verts: {:0.4f}'.format(
        n_eval_data_times, np.mean(error_norm_dl)))
    print('Linear method: error norm average for {} verts: {:0.4f}\n'.format(
        n_eval_data_times, np.mean(error_norm_mne)))
    print('Deep learning; Accuracy: {:0.5f}, Avg. Point spread: {:0.5f}'.format(
        accuracy_dl, np.mean(point_spread_dl)))
    print('Linear method; Accuracy: {:0.5f}, Avg. Point spread: {:0.5f}\n'.format(
        accuracy_mne, np.mean(point_spread_mne)))
