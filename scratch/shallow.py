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

    print("  %s: -- finished loading meg objects" % subj)

    return subj, fwd, inv, cov, evoked_info


def gen_evoked_subject(signal, fwd, cov, evoked_info, dt, noise_snr,
                       seed=None):
    """Function to generate evoked and stc from signal array"""

    vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc = SourceEstimate(signal, vertices, tmin=0, tstep=dt)

    evoked = simulate_evoked(fwd, stc, evoked_info, cov, noise_snr,
                             random_state=seed)
    evoked.add_eeg_average_proj()

    return evoked, stc


def get_data_batch(x_data, y_label, batch_num, batch_size):
    """Function to get a random sampling of an evoked and stc pair"""

    # Get random sampling of data, seed by batch num
    np.random.seed(batch_num)
    rand_inds = np.random.randint(x_data.shape[0], size=batch_size)

    return x_data[rand_inds, :], y_label[rand_inds, :]


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.11)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def make_tanh_network(sensor_dim, source_dim):
    """Function to create neural network"""

    x_sensor = tf.placeholder(tf.float32, shape=[None, sensor_dim], name="x_sensor")

    W1 = weight_variable([sensor_dim, source_dim // 2])
    b1 = bias_variable([source_dim // 2])
    h1 = tf.nn.tanh(tf.matmul(x_sensor, W1) + b1)

    W2 = weight_variable([source_dim // 2, source_dim // 2])
    b2 = bias_variable([source_dim // 2])
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

    W3 = weight_variable([source_dim // 2, source_dim])
    b3 = bias_variable([source_dim])
    yhat = tf.nn.tanh(tf.matmul(h2, W3) + b3)

    return yhat, h1, h2, x_sensor


def sparse_objective(sensor_dim, source_dim, yhat, h1, h2, sess):

    y_source = tf.placeholder(tf.float32, shape=[None, source_dim], name="y_source")
    rho = tf.placeholder(tf.float32, shape=(), name="rho")
    lam = tf.placeholder(tf.float32, shape=(), name="lam")

    diff = y_source - yhat
    error = tf.reduce_sum(tf.squared_difference(y_source, yhat))

    # Remap activations to [0,1]
    a1 = 0.5 * h1 + 0.5
    a2 = 0.5 * h2 + 0.5

    kl_bernoulli_h1 = (rho*(tf.log(rho) - tf.log(a1+1e-6)
                      + (1-rho)*(tf.log(1-rho) - tf.log(1-a1+1e-6))))
    kl_bernoulli_h2 = (rho*(tf.log(rho) - tf.log(a2+1e-6)
                      + (1-rho)*(tf.log(1-rho) - tf.log(1-a2+1e-6))))
    regularizer = (tf.reduce_sum(kl_bernoulli_h1)
                   + tf.reduce_sum(kl_bernoulli_h2))

    cost = error + lam*regularizer

    return cost, y_source, rho, lam


def eval_error_norm(src_data_orig, src_data_est):
    """Function to compute norm of the error vector at each dipoe

    Parameters
    ----------

    src_data_orig: numpy matrix size (n_samples x n_src)
        Ground truth source estimate used to generate sensor data

    src_data_est: numpy matrix (n_samples x n_src)
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

    Parameters
    ----------
    true_pos: np.array shape(n_times, 3)
        3D position of dipole that was simulated active
    largest_dip_pos: np.array shape(n_times, n_dipoles, 3)
        3D positions of top `n_dipoles` dipoles with highest activation"""

    centroids = np.mean(largest_dip_pos, axis=1)
    accuracy = np.linalg.norm(true_pos - centroids)

    # Calculate difference in x/y/z positions from true activation to each src
    point_distance = np.subtract(largest_dip_pos, true_pos[:, np.newaxis, :])

    # Calculate Euclidean distance (w/ norm) and take mean over all dipoles
    point_spread = np.mean(np.linalg.norm(point_distance, axis=-1), axis=-1)

    return accuracy, point_spread

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

    # Set params
    n_data_times = 50000
    dt = 0.001
    noise_snr = np.inf
    batch_size = 1000

    n_iter = int(100000)
    rho = 0.05
    lam = 1.

    # Get subject info and create data
    subj, fwd, inv, cov, evoked_info = load_subject_objects(megdir, subj,
                                                            struct)
    n_verts = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    sim_vert_data = np.random.randn(n_verts, n_data_times)

    print("Simulating and normalizing data")
    evoked, stc = gen_evoked_subject(sim_vert_data, fwd, cov, evoked_info, dt,
                                     noise_snr)
    # Normalize data to lie between -1 and 1
    x_sens_all = np.ascontiguousarray(evoked.data.T)
    x_sens_all /= np.max(np.abs(x_sens_all))
    y_src_all = np.ascontiguousarray(sim_vert_data.T)
    y_src_all /= np.max(np.abs(y_src_all))

    print("Building neural net")
    sess = tf.Session()

    sensor_dim = evoked.data.shape[0]
    source_dim = n_verts

    yhat, h1, h2, x_sensor = make_tanh_network(sensor_dim, source_dim)
    sparse_cost, y_source, tf_rho, tf_lam = sparse_objective(sensor_dim,
                                                             source_dim, yhat,
                                                             h1, h2, sess)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(sparse_cost)

    sess.run(tf.initialize_all_variables())
    print("\nSim params\n----------\nn_iter: {}\nn_data_points: {}\nSNR: \
          {}\nbatch_size: {}\n".format(n_iter, n_data_times, str(noise_snr),
                                       batch_size))
    print("Optimizing...")
    for ii in xrange(n_iter):
        # Get random batch of data
        x_sens_batch, y_src_batch = get_data_batch(x_sens_all, y_src_all, ii,
                                                   batch_size)

        feed_dict = {x_sensor: x_sens_batch, y_source: y_src_batch,
                     tf_rho: rho, tf_lam: lam}
        _, obj = sess.run([train_step, sparse_cost], feed_dict)

        if ii % 10 == 0:
            print("\titer: %d, cost: %.2f" % (ii, obj))

    #
    # Evaluate network
    #
    n_eval_data_times = 1000  # Probably should be <= 1000 to avoid mem problems

    print("\nEvaluating...\n")

    # Simulate identity activations
    rand_verts = np.sort(np.random.randint(0, n_verts, n_eval_data_times))
    sim_vert_eval_data = np.eye(n_verts)[:, rand_verts] * 0.1
    evoked_eval, stc_eval = gen_evoked_subject(sim_vert_eval_data, fwd, cov,
                                               evoked_info, dt, noise_snr)

    x_sens_eval = np.ascontiguousarray(evoked_eval.data.T)
    y_src_eval = np.ascontiguousarray(stc_eval.data.T)

    feed_dict = {x_sensor: x_sens_eval, y_source: y_src_eval,
                 tf_rho: rho, tf_lam: lam}
    src_est = sess.run(yhat, feed_dict)

    # Calculate vector norm error
    error_norm = eval_error_norm(y_src_eval, src_est)

    dip_positions = get_all_vert_positions(inv['src'])

    # Ground truth pos
    activation_positions = dip_positions[rand_verts, :]
    # Get position of most active dipoles
    largest_dip_positions = get_largest_dip_positions(src_est, 25,
                                                      dip_positions)
    # Calculate accuracy metrics (in meters)
    accuracy, point_spread = get_localization_metrics(activation_positions,
                                                      largest_dip_positions)
