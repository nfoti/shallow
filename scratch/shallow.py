""" shallow.py

    Run deep neural net on sensor space signals from known source space
    signals.

    Usage:
      shallow.py <megdir> <structdir> [--subj=<subj>]
      shallow.py (-h | --help)

    Options:
      --subj=<subj>     Specify single subject to process with structural name
      -h, --help        Show this screen
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as op
import numpy as np
from time import strftime, sleep
import pprint
from shutil import copy2
from copy import deepcopy

import keras
from keras.callbacks import BaseLogger, TensorBoard, ModelCheckpoint
from keras.utils.visualize_util import plot as keras_plot

import mne
from mne import SourceEstimate
from mne.minimum_norm import read_inverse_operator
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked
from mne.externals.h5io import read_hdf5, write_hdf5

import config_exp
from config_exp import training_params as TP
from shallow_fun import (construct_model, load_model_specs, norm_transpose,
                         gen_subject_train_data, load_subject_objects)

shallow_dir = os.environ['SHALLOW_DIR']

structs = config_exp.structurals[0:10]
subjects = config_exp.subjects[0:10]


def create_exp_save_dir(exp_dir, n_subjects, n_models):
    """Helper to get/create save directory for one experiment"""
    if not op.isdir(exp_dir):
        os.mkdir(exp_dir)

    # Create folder with current time to save experimental results
    fold_name_cur_exp = strftime('%Y_%m_%d__%H_%M_') + \
        '%i_subj_%i_models' % (n_subjects, n_models)
    dir_name_cur_exp = op.join(exp_dir, fold_name_cur_exp)
    os.mkdir(dir_name_cur_exp)

    return dir_name_cur_exp


def create_subj_save_dir(exp_save_dir, subj):
    """Helper to get/create save directory for one subject"""
    dir_name_cur_subj = op.join(exp_save_dir, subj)
    if not op.isdir(dir_name_cur_subj):
        os.mkdir(dir_name_cur_subj)

    return dir_name_cur_subj


if __name__ == "__main__":

    # Gather command line params from docopt
    from docopt import docopt
    argv = docopt(__doc__)

    megdir = argv['<megdir>']
    structdir = argv['<structdir>']

    if argv['--subj']:
        structs = [argv['--subj']]
        subjects = [subjects[structs.index(structs[0])]]

    print('Starting training %s' % strftime('%m/%d %H:%M:%S'))
    ############################################
    # Load config info and create save dir
    ############################################
    n_training_samps = TP['n_training_samps_noise']
    n_valid_samps = TP['n_training_samps_noise'] * TP['valid_proportion']
    model_specs = load_model_specs('config_model.yaml')

    # Create folder for saving current experiment, copy config files over
    exp_save_dir = create_exp_save_dir(
        op.join(shallow_dir, config_exp.exp_fold), len(subjects), len(model_specs))
    copy2('config_exp.py', op.join(exp_save_dir, 'config_exp_at_training.py'))
    copy2('config_model.yaml', exp_save_dir)

    pp = pprint.PrettyPrinter(indent=4)
    print('\nSubjects: (%i)' % len(subjects))
    pp.pprint(subjects)
    print('\nModels: (%i)' % len(model_specs))
    pp.pprint(model_specs)
    print('\nParams:')
    pp.pprint(TP)

    ############################################
    # Loop over subjects
    ############################################
    for si, (subj, struct) in enumerate(zip(subjects, structs)):
        subj_save_dir = create_subj_save_dir(exp_save_dir, struct)

        # Define Keras callbacks
        callbacks = [BaseLogger(),
                     ModelCheckpoint(op.join(subj_save_dir, 'saved_model_' +
                                             '{epoch:04d}_{val_loss:.3f}.hdf5'),
                                     save_best_only=True),
                     TensorBoard(log_dir=subj_save_dir)]

        ############################################
        # Load subject head models and generate data
        ############################################
        fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj, struct)
        sen_dim = len(fwd['info']['ch_names'])
        src_dim = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

        # XXX: Consider moving to a keras `fit_generator` approach so that new
        # data is generated for each epoch

        # Generating random brain activity and passing through fwd solution
        sim_src_act = np.random.randn(src_dim, TP['n_training_samps_noise'])
        sim_src_act *= TP['src_act_scaler']

        train_x, train_y, _ = gen_subject_train_data(
            sim_src_act, fwd, cov, evoked_info, TP['dt'], TP['SNR'])

        ############################################
        # Train and save neural network
        ############################################

        # Loop over each model configuration in gridsearch
        for mi, model_spec in enumerate(model_specs):
            model = construct_model(model_spec, sen_dim, src_dim)

            # Gather all model specs and print
            print('\nSubj %i/%i Model %i/%i)' % (si + 1, len(subjects),
                                                 mi + 1, len(model_specs)))
            temp_spec = deepcopy(model_spec)
            temp_spec['arch'].insert(0, sen_dim)
            temp_spec['arch'][-1] = src_dim
            pp.pprint(temp_spec)

            # Use separate logdir for each model
            if type(callbacks[2]) is keras.callbacks.TensorBoard:
                temp_tb_logdir = op.join(subj_save_dir, 'tb_%i' % mi)
                os.mkdir(temp_tb_logdir)
                callbacks[2].log_dir = temp_tb_logdir

            history = model.fit(train_x, train_y,
                                nb_epoch=model_spec['n_epochs'],
                                batch_size=model_spec['batch_size'],
                                validation_split=TP['valid_proportion'],
                                callbacks=callbacks,
                                verbose=2)

            sleep(1)  # Try to deal with delayed garbage collection

            # Save model architecture to picture
            if si is 0:
                keras_plot(model, show_shapes=True,
                           to_file=op.join(exp_save_dir, 'model_%i.png' % mi))
    print('Finished training %s' % strftime('%m/%d %H:%M:%S'))
