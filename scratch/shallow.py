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
#from vae import nnet


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


def make_network():
    """Create neural network"""
    net = None
    return net


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

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    n_verts = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    sim_vert_data = np.random.randn(n_verts, n_times)

    evoked, stc = gen_evoked_subject(sim_vert_data, fwd, cov, evoked_info, dt,
                                     noise_snr)

    # Create neural network
    # net = make_net()

    # Train net
    #   Make data iterator for tensorflow
    #   Save checkpoints after training each batch
    #   Visualize training loss

    # Evaluate net
