"""eval_model.py

    Evaluate deep neural net on sensor space signals from known source space
    signals. Tests localization error and point spread

    Usage:
      eval_model.py <megdir> <structdir> [--subj=<subj>]
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
from shutil import copy2
import yaml
from itertools import product

import mne
from mne.simulation import simulate_sparse_stc, simulate_stc
from mne.minimum_norm import apply_inverse

import keras
from keras.models import load_model
import numpy as np

from shallow_fun import (load_subject_objects, gen_subject_train_data,
                         get_all_vert_positions, get_largest_dip_positions,
                         get_localization_metrics, eval_error_norm,
                         norm_transpose, load_model_specs)
from config_exp import structurals, subjects
from config_exp import training_params as TP
from config_exp import eval_params as EP

# Get subjects and file names
structurals = structurals[0:1]
subjects = subjects[0:1]

shallow_dir = os.environ['SHALLOW_DIR']
exp_name = '2016_12_12__14_41_2_subj_4_models'
exp_save_dir = op.join(shallow_dir, 'exp_results', exp_name)


def create_subj_eval_dir(exp_save_dir, subj):
    """Helper to get/create save director for one subject"""
    dir_name_cur_subj = op.join(exp_save_dir, subj)
    if not op.isdir(dir_name_cur_subj):
        os.mkdir(dir_name_cur_subj)
    return dir_name_cur_subj


def get_model_ind(model_spec, full_specs):
    """Helper to get an array index from a dict of keys"""

    return tuple([full_specs[key].index(model_spec[key])
                  for key in full_specs])


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

    ###################################
    # Loop over subjects
    ###################################
    for si, (subj, struct) in enumerate(zip(subjects, structurals)):
        # Get folder with subject's saved models/info
        subj_results_dir = create_subj_eval_dir(exp_save_dir, struct)
        # Archive current config file
        copy2('config_exp.py', op.join(exp_save_dir, 'config_exp_at_eval.py'))

        # Get filenames of all saved models
        subj_models = [tmp_fname for tmp_fname in os.listdir(subj_results_dir)
                       if 'saved_model_' in tmp_fname]
        subj_models.sort()

        # Get configuration file, load model specifications and init scores
        model_conf_file = op.join(exp_save_dir, 'config_model.yaml')
        with open(model_conf_file, 'rb') as yaml_file:
            model_params = yaml.load(yaml_file)
            metrics = model_params.pop('metrics')
        n_score_methods = len(EP['score_methods'])
        scores_arr_shape = np.array([len(tp) for tp in model_params.values()] +
                                    [n_score_methods])
        scores_dl = np.zeros(scores_arr_shape)

        model_specs = load_model_specs(model_conf_file)

        # Get subject info and create data
        fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj, struct)
        vert_list = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        n_verts = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

        sen_dim = len(fwd['info']['ch_names'])
        src_dim = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

        #########################################
        # Simulate sparse unit dipole activations
        #########################################

        # Construct data for localizing activity
        n_avg_verts = EP['n_avg_verts']
        #n_avg_verts = n_verts
        n_test_verts = EP['n_test_verts']
        rand_verts = np.sort(np.random.choice(range(n_verts), n_test_verts,
                                              replace=False))
        sim_src_data = np.eye(n_verts)[:, rand_verts]
        test_x_sparse, test_y_sparse, evo_sparse = gen_subject_train_data(
            sim_src_data, fwd, cov, evoked_info, TP['dt'], TP['SNR'])
        # Ground truth dipole positions
        vert_positions = get_all_vert_positions(inv['src'])
        true_act_positions = vert_positions[rand_verts, :]

        #########################################
        # Simulate dense dipole activations
        #########################################

        # Construct data for estimating random activity across entire src space
        sim_src_data = np.random.randn(src_dim, TP['n_training_samps_noise'])
        sim_src_data *= TP['src_act_scaler']
        test_x_dense, test_y_dense, evo_dense = gen_subject_train_data(
            sim_src_data, fwd, cov, evoked_info, TP['dt'], TP['SNR'])

        #############################################
        # Evaluate model on sparse and dense activity
        #############################################
        print("\nEvaluating saved models...\n")
        for mi, model_fname in enumerate(subj_models):
            model = load_model(op.join(exp_save_dir, struct, model_fname))

            # Get position of most active dipoles and calc metrics (in meters)
            pred_y_sparse = model.predict(test_x_sparse, batch_size=25)
            est_act_positions = get_largest_dip_positions(pred_y_sparse,
                                                          n_avg_verts,
                                                          vert_positions)

            #stc_dl = SourceEstimate(pred_y_sparse.T, vertices=vert_list,
            #                        subject=struct, tmin=0,
            #                        tstep=common_params['dt'])

            loc_accuracy_dl, point_spread_dl = get_localization_metrics(
                true_act_positions, est_act_positions)

            # Calc performance of estimating dense random activity
            pred_y_dense = model.predict(test_x_dense, batch_size=25)
            error_norm_dl = eval_error_norm(test_y_dense, pred_y_dense)

            # Store scores
            ind = get_model_ind(model_specs[mi], model_params)
            scores_dl[ind] = [loc_accuracy_dl, np.mean(point_spread_dl),
                              np.mean(error_norm_dl)]

            print('\tModel %03i. Loc_acc: %03.3f Point spread: %03.3f l2 error: %03.3f' %
                  (mi, loc_accuracy_dl, np.mean(point_spread_dl),
                   np.mean(error_norm_dl)))

        #############################################
        # Evaluate standard MNE methods
        #############################################
        print("\nEvaluating standard linear approach...\n")

        # Get position of most active dipoles and calc metrics (in meters)
        pred_stc_sparse = apply_inverse(evo_sparse, inv,
                                        method=EP['linear_inv'])
        pred_y_sparse = pred_stc_sparse.data.T

        est_act_positions = get_largest_dip_positions(pred_y_sparse,
                                                      n_avg_verts,
                                                      vert_positions)
        loc_accuracy_std, point_spread_std = get_localization_metrics(
            true_act_positions, est_act_positions)

        # Calc performance of estimating dense random activity
        pred_stc_dense = apply_inverse(evo_sparse, inv, method=EP['linear_inv'])
        pred_y_dense = pred_stc_dense.data.T

        error_norm_std = eval_error_norm(test_y_dense, pred_y_dense)

        scores_std = np.array([loc_accuracy_std, np.mean(point_spread_std),
                               np.mean(error_norm_dl)])
        print('\tModel std. Loc_acc: %03.3f Point spread: %03.3f l2 error: '
              '%03.3f' % (loc_accuracy_std, np.mean(point_spread_std),
                          np.mean(error_norm_dl)))

        # Save all scores
        fname_scores = op.join(exp_save_dir, struct, 'model_scores.npz')
        np.savez(fname_scores, scores_dl=scores_dl, scores_std=scores_std)
