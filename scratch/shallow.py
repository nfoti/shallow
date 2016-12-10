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
from time import strftime
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
from shallow_fun import construct_model, load_model_specs, norm_transpose

shallow_dir = os.environ['SHALLOW_DIR']

# Removing eric_sps_32/AKCL_132 b/c of vertex issue
structs = config_exp.structurals[:5]
subjects = config_exp.subjects[:5]


def load_subject_objects(megdatadir, subj, struct):

    print("  %s: -- loading meg objects" % subj)

    fname_fwd = op.join(megdatadir, subj, 'forward',
                        '%s-sss-fwd.fif' % subj)
    fwd = mne.read_forward_solution(fname_fwd, force_fixed=True, surf_ori=True,
                                    verbose=False)

    fname_inv = op.join(megdatadir, subj, 'inverse',
                        '%s-55-sss-meg-eeg-fixed-inv.fif' % subj)
    inv = read_inverse_operator(fname_inv, verbose=False)

    fname_epochs = op.join(megdatadir, subj, 'epochs',
                           'All_55-sss_%s-epo.fif' % subj)
    #epochs = mne.read_epochs(fname_epochs)
    #evoked = epochs.average()
    #evoked_info = evoked.info
    evoked_info = mne.io.read_info(fname_epochs, verbose=False)
    cov = inv['noise_cov']

    print("  %s: -- finished loading meg objects" % subj)

    return fwd, inv, cov, evoked_info


def gen_evoked_subject(signal, fwd, cov, evoked_info, dt, noise_snr,
                       seed=None):
    """Function to generate evoked and stc from signal array"""

    vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc = SourceEstimate(signal, vertices, tmin=0, tstep=dt, verbose=False)

    evoked = simulate_evoked(fwd, stc, evoked_info, cov, noise_snr,
                             random_state=seed, verbose=False)
    evoked.set_eeg_reference()

    # Normalize training data to lie between -1 and 1
    # XXX: Appropriate to do this? See batch normalization papers
    x_train = norm_transpose(evoked.data)
    y_train = norm_transpose(signal)

    return x_train, y_train


def create_exp_save_dir(exp_dir, n_subjects, n_models):
    """Helper to get/create save director for one experiment"""
    if not op.isdir(exp_dir):
        os.mkdir(exp_dir)

    # Create folder to save current experiment
    fold_name_cur_exp = strftime('%Y_%m_%d__%H_%M_') + \
        '%i_subj_%i_models' % (n_subjects, n_models)
    dir_name_cur_exp = op.join(exp_dir, fold_name_cur_exp)
    os.mkdir(dir_name_cur_exp)

    return dir_name_cur_exp


def create_subj_save_dir(exp_save_dir, subj):
    """Helper to get/create save director for one subject"""
    dir_name_cur_subj = op.join(exp_save_dir, subj)
    if not op.isdir(dir_name_cur_subj):
        os.mkdir(dir_name_cur_subj)
    return dir_name_cur_subj


"""
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
"""


if __name__ == "__main__":

    from docopt import docopt
    argv = docopt(__doc__)

    megdir = argv['<megdir>']
    structdir = argv['<structdir>']

    if argv['--subj']:
        structs = [argv['--subj']]
        subjects = [subjects[structs.index(structs[0])]]

    ############################################
    # Load config info and create save dir
    ############################################
    n_training_samps = TP['n_training_samps_noise']
    n_valid_samps = TP['n_training_samps_noise'] * TP['valid_proportion']
    model_specs = load_model_specs('config_model.yaml')

    # Create folder for saving current experiment, copy over config files
    exp_save_dir = create_exp_save_dir(
        op.join(shallow_dir, config_exp.exp_fold), len(subjects), len(model_specs))
    copy2('config_exp.py', exp_save_dir)
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

        # Define callbacks
        callbacks = [BaseLogger(),
                     ModelCheckpoint(op.join(subj_save_dir, 'saved_model_' + \
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

        sim_src_act = np.random.randn(src_dim, TP['n_training_samps_noise'])
        sim_src_act *= TP['src_act_scaler']

        train_x, train_y = gen_evoked_subject(sim_src_act, fwd, cov,
                                              evoked_info, TP['dt'], TP['SNR'])
        '''
        def subj_spec_generator(batch_size, epoch_size):
            while 1:
                sim_src_act = np.random.randn(src_dim,
                                            TP['n_training_samps_noise']) * \
                    TP['src_act_scaler']

                train_x, train_y = gen_evoked_subject(sim_src_act, fwd, cov,
                                                      evoked_info, TP['dt'],
                                                      TP['SNR'])
                i = 0
                while i < epoch_size:
                    yield (train_x[i:i + batch_size], train_y[i:i + batch_size])
                    if i + batch_size > epoch_size:
                        i = 0
                    else:
                        i += batch_size
        '''

        ############################################
        # Train neural network
        ############################################

        for mi, model_spec in enumerate(model_specs):
            model = construct_model(model_spec, sen_dim, src_dim)

            # Print model information
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
                                callbacks=callbacks)

            '''
            model.fit_generator(subj_spec_generator(model_spec['batch_size'],
                                                    TP['n_training_samps_noise']),
                                samples_per_epoch=TP['n_training_samps_noise'],
                                nb_epoch=model_spec['n_epochs'],
                                validation_data=subj_spec_generator(
                                    model_spec['batch_size'],
                                    TP['n_training_samps_noise']),
                                nb_val_samples=n_valid_samps,
                                callbacks=callbacks,
                                max_q_size=1)
            '''

            if si is 0:
                keras_plot(model, show_shapes=True,
                           to_file=op.join(exp_save_dir, 'model_%i.png' % mi))

        ############################################
        # Evaluate net
        ############################################
