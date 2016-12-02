"""
config_exp.py

@author: wronk

Configuration file for training/testing shallow inverse model
"""
import numpy as np

# Entries in structurals and subjects must correspoond,
# i.e. structurals[i] === subjects[i].
# 'AKCLEE_132' remove because of vertex issues
structurals = ['AKCLEE_103', 'AKCLEE_104', 'AKCLEE_105', 'AKCLEE_106',
               'AKCLEE_107', 'AKCLEE_109', 'AKCLEE_110', 'AKCLEE_115',
               'AKCLEE_117', 'AKCLEE_118', 'AKCLEE_119', 'AKCLEE_121',
               'AKCLEE_125', 'AKCLEE_126', 'AKCLEE_131']
subjects = ['eric_sps_03', 'eric_sps_04', 'eric_sps_05', 'eric_sps_06',
            'eric_sps_07', 'eric_sps_09', 'eric_sps_10', 'eric_sps_15',
            'eric_sps_17', 'eric_sps_18', 'eric_sps_19', 'eric_sps_21',
            'eric_sps_25', 'eric_sps_26', 'eric_sps_31']

exp_fold = 'exp_results'

# Model training params
training_params = dict(dt=0.001,
                       SNR=np.inf,
                       n_training_samps_noise=1e6,  # Number of noise data samples
                       n_training_samps_sparse=0,  # Number of sparse data samples
                       src_act_scaler=1e-7,
                       batch_size=100,
                       valid_proportion=0.2)

# Model evaluation params
eval_params = dict(n_avg_verts=25,  # Number of verts to avg when determining est position
                   n_test_verts=1000,  # Probably should be <= 1000 to avoid mem problems
                   linear_inv='MNE')  # sLORETA or MNE
