"""
config.py

@author: wronk

Configuration file for training/testing shallow inverse model
"""

import numpy as np

# Entries in structurals and subjects must correspoond,
# i.e. structurals[i] === subjects[i].

# Structural MRI subject names
conf_structurals = ['AKCLEE_103', 'AKCLEE_104', 'AKCLEE_105', 'AKCLEE_106',
                    'AKCLEE_107', 'AKCLEE_109', 'AKCLEE_110', 'AKCLEE_115',
                    'AKCLEE_117', 'AKCLEE_118', 'AKCLEE_119', 'AKCLEE_121',
                    'AKCLEE_125', 'AKCLEE_126', 'AKCLEE_131', 'AKCLEE_132']

# Experimental subject names
conf_subjects = ['eric_sps_03', 'eric_sps_04', 'eric_sps_05', 'eric_sps_06',
                 'eric_sps_07', 'eric_sps_09', 'eric_sps_10', 'eric_sps_15',
                 'eric_sps_17', 'eric_sps_18', 'eric_sps_19', 'eric_sps_21',
                 'eric_sps_25', 'eric_sps_26', 'eric_sps_31', 'eric_sps_32']

# Model params for training/testing
common_params = dict(dt=0.001,
                     SNR=np.inf,
                     rho=0.1,
                     lam=1.)  # Weighting of regularizer cost

# Model training params
training_params = dict(n_training_times_noise=1000,  # Number of noise data samples
                       n_training_times_sparse=0,  # Number of sparse data samples
                       batch_size=100,
                       n_training_iters=int(1e4),  # Number of training iterations
                       opt_lr=1e-4)  # Learning rate for optimizer

# Model evaluation params
eval_params = dict(n_avg_verts=25,  # Number of verts to avg when determining est position
                   n_test_verts=1000,  # Probably should be <= 1000 to avoid mem problems
                   linear_inv='MNE')  #sLORETA or MNE
