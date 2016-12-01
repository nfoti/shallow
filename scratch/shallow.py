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
import numpy as np

import mne
from mne import SourceEstimate
from mne.minimum_norm import read_inverse_operator
from mne.simulation import simulate_sparse_stc, simulate_stc, simulate_evoked
from mne.externals.h5io import read_hdf5, write_hdf5

import config_exp
from config_exp import training_params as TP
from shallow_fun import construct_model, load_model_specs, norm_transpose


# Removing eric_sps_32/AKCL_132 b/c of vertex issue
structurals = config_exp.structurals[:-1]
subjects = config_exp.subjects[:-1]


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
    stc = SourceEstimate(signal, vertices, tmin=0, tstep=dt)

    evoked = simulate_evoked(fwd, stc, evoked_info, cov, noise_snr,
                             random_state=seed)
    evoked.set_eeg_reference()

    return evoked, stc


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

    struct = None
    subj = None
    if argv['--subj']:
        struct = argv['--subj']
        subj = subjects[structurals.index(struct)]

    ############################################
    # Load subject head models and generate data
    ############################################

    n_training_times = TP['n_training_times_noise']
    n_training_iters = TP['n_training_iters']

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    sen_dim = len(fwd['info']['ch_names'])
    src_dim = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

    print("Simulating data")
    sim_src_act = np.random.randn(src_dim, TP['n_training_times_noise']) * \
        TP['src_act_scaler']
    evoked, stc = gen_evoked_subject(sim_src_act, fwd, cov, evoked_info,
                                     TP['dt'], TP['SNR'])

    # Normalize training data to lie between -1 and 1
    # XXX: Appropriate to do this? Maybe need to normalize src space only
    # before generating sens data
    x_train = norm_transpose(evoked.data)
    y_train = norm_transpose(sim_src_act)

    x_sens = np.ascontiguousarray(evoked.data.T)
    x_sens /= np.max(np.abs(x_sens))
    y_src = np.ascontiguousarray(sim_src_act.T)
    y_src /= np.max(np.abs(y_src))

    ############################################
    # Create neural network
    ############################################

    print("Training...")

    model_specs = load_model_specs('config_model.yaml')
    for model_spec in model_specs:
        model = construct_model(model_spec, sen_dim)
        model.fit(x_train, y_train,
                  nb_epoch=model_spec['n_epochs'],
                  batch_size=TP['batch_size'],
                  validation_split=TP['valid_proportion'])

    ############################################
    # Evaluate net
    ############################################
