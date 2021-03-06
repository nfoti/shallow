"""
shallow_fun.py

@author: wronk

Various functions useful for shallow
"""
from itertools import product
import os.path as op
import yaml
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import WeightRegularizer as weight_reg
from keras.regularizers import ActivityRegularizer as act_reg

import mne
from mne import SourceEstimate
from mne.minimum_norm import read_inverse_operator
from mne.simulation import simulate_evoked


def load_model_specs(fname_yaml_conf):
    """
    Loads yaml file and converts it into list of model specification dicts
    """

    with open(fname_yaml_conf, 'rb') as yaml_file:
        model_specs = yaml.load(yaml_file)
        metrics = model_specs.pop('metrics')

    params_list = list(product(*model_specs.values()))

    model_spec_list = []
    for temp_specs in params_list:
        model_spec_list.append(dict(zip(model_specs.keys(), temp_specs)))
        model_spec_list[-1]['metrics'] = metrics

    return model_spec_list


def gen_subject_train_data(signal, fwd, cov, evoked_info, dt, noise_snr,
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
    evoked.data = x_train.T

    return x_train, y_train, evoked


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


def construct_model(model_spec, input_dim, output_dim):
    """
    Helper to construct a Keras model based on dict of specs and input size

    Parameters
    ----------
    model_spec: dict
        Dict containing keys: arch, activation, dropout, optimizer, loss,
            w_reg, metrics
    input_dim: int
        Size of input dimension
    output_dim: int
        Size of input dimension

    Returns
    -------
    model: Compiled keras.models.Sequential

    """

    model = Sequential()

    for li, layer_size in enumerate(model_spec['arch']):
        # Set output size for last layer
        if layer_size == 'None':
            layer_size = output_dim

        # For input layer, add input dimension
        if li == 0:
            temp_input_dim = input_dim
            model.add(Dense(layer_size,
                            input_dim=input_dim,
                            activation=model_spec['activation'],
                            W_regularizer=weight_reg(model_spec['w_reg'][0],
                                                     model_spec['w_reg'][1]),
                            name='Input'))
        else:
            model.add(Dense(layer_size,
                            activation=model_spec['activation'],
                            W_regularizer=weight_reg(model_spec['w_reg'][0],
                                                     model_spec['w_reg'][1]),
                            name='Layer_%i' % li))

        if model_spec['dropout'] > 0.:
            model.add(Dropout(model_spec['dropout'], name='Dropout_%i' % li))

    model.compile(optimizer=model_spec['optimizer'],
                  loss=model_spec['loss'],
                  metrics=model_spec['metrics'])

    return model


def norm_transpose(data):
    """Helper to transpose data and normalize by max val of each row"""
    #XXX Probably should switch to sklearn's standard scaler
        # (0 mean, unit var, and saves the scaling if we need to apply it later)
    data_fixed = np.ascontiguousarray(data.T)
    data_fixed /= np.max(np.abs(data_fixed), axis=0)

    return data_fixed


def get_all_vert_positions(src):
    """Function to get 3-space position of used dipoles

    Parameters
    ----------
    src: SourceSpaces
        Source space object for subject. Needed to get dipole positions

    Returns
    -------
    dip_pos: np.array shape(n_src x 3)
        3-space positions of used dipoles
    """
    # Get vertex numbers and positions that are in use
    # (usually ~4k - 5k out of ~150k)
    left_vertno = src[0]['vertno']
    right_vertno = src[1]['vertno']

    vertnos = np.concatenate((left_vertno, right_vertno))
    dip_pos = np.concatenate((src[0]['rr'][left_vertno, :],
                              src[1]['rr'][right_vertno, :]))

    return dip_pos


def eval_error_norm(src_data_orig, src_data_est):
    """Function to compute norm of the error vector at each dipole

    Parameters
    ----------

    src_data_orig: np.array size (n_samples x n_src)
        Ground truth source estimate used to generate sensor data

    src_data_est: np.array size (n_samples x n_src)
        Source estimate of sensor data created using src_data_orig

    Returns
    -------
    error_norm: np.array size(n_samples)
        Norm of vector between true activation and estimated activation

    """

    #TODO: might want to normalize by number of vertices since subject source
    #      spaces can have different number of dipoles

    error_norm = np.zeros((src_data_orig.shape[0]))

    for ri, (row_orig, row_est) in enumerate(zip(src_data_orig, src_data_est)):
        error_norm[ri] = np.linalg.norm(row_orig - row_est)

    return error_norm


def get_largest_dip_positions(data, n_verts, dip_pos):
    """Function to get spatial centroid of highest activated dipoles

    Parameters
    ----------
    data: np.array shape(n_times x n_src)
        Source estimate data
    n_verts: int
        Number of vertices to use when computing maximum activation centroid
    dip_pos: np.array shape(n_src x 3)
        3-space positions of all dipoles in source space .

    Returns
    -------
    avg_pos: np.array shape(n_times x 3)
        Euclidean centroid of activation for largest `n_verts` activations
    """

    #TODO: How to handle negative current vals? Use abs?

    # Initialize
    largest_dip_pos = np.zeros((data.shape[0], n_verts, 3))

    # Find largest `n_verts` dipoles at each time point and get position
    for ti in range(data.shape[0]):
        largest_dip_inds = data[ti, :].argsort()[-n_verts:]
        largest_dip_pos[ti, :, :] = dip_pos[largest_dip_inds, :]

    return largest_dip_pos


def get_localization_metrics(true_pos, largest_dip_pos):
    """Helper to get accuracy and point spread

    Accuracy: Euclidean distance between centroid of top n dipoles and the true
        center of activation.
    Point Spread: Average Euclidean distance between each of the top n dipoles
        and the true center of activation.

    Parameters
    ----------
    true_pos: np.array shape(n_times, 3)
        3D position of dipole that was simulated active
    largest_dip_pos: np.array shape(n_times, n_dipoles, 3)
        3D positions of top `n_dipoles` dipoles with highest activation

    Returns
    -------
    accuracy: float
        Euclidean distance between true position and centroid of estimated
        activity

    point_spread: np.array shape(n_times,)
        Average Euclidean distance between true activity position and each of
        the top n active dipoles
    """

    centroids = np.mean(largest_dip_pos, axis=1)
    accuracy = np.linalg.norm(true_pos - centroids)

    # Calculate difference in x/y/z positions from true activation to each src
    point_distance = np.subtract(largest_dip_pos, true_pos[:, np.newaxis, :])

    # Calculate Euclidean distance (w/ norm) and take mean over all dipoles
    point_spread = np.mean(np.linalg.norm(point_distance, axis=-1), axis=-1)

    return accuracy, point_spread
