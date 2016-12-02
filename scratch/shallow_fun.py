"""
shallow_fun.py

@author: wronk

Various functions useful for shallow
"""
from itertools import product
import yaml
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


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


def construct_model(model_spec, input_dim, output_dim):
    """
    Helper to construct a Keras model based on dict of specs and input size

    Parameters
    ----------
    model_spec: dict
        Dict containing keys: arch, activation, dropout, optimizer, loss,
            metrics
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
            model.add(Dense(layer_size, input_dim=input_dim,
                            activation=model_spec['activation'],
                            name='Input'))
        else:
            model.add(Dense(layer_size, activation=model_spec['activation'],
                            name='Layer_%i' % li))

        if model_spec['dropout'] > 0.:
            model.add(Dropout(model_spec['dropout']))

    model.compile(optimizer=model_spec['optimizer'],
                  loss=model_spec['loss'],
                  metrics=model_spec['metrics'])

    return model


def norm_transpose(data):
    """Helper to transpose data and normalize by max val"""
    #XXX Probably should switch to sklearn's standard scaler
        # (0 mean, unit var, and saves the scaling if we need to apply it later)
    data_fixed = np.ascontiguousarray(data.T)
    data_fixed /= np.max(np.abs(data_fixed))

    return data_fixed
