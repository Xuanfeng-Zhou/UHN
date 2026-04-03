'''
Configuration file for base network configurations
'''
from typing import Dict, List, Tuple
from model.model.model import TaskType, ModelType, DatasetType, FEATURES_DICT, OUTPUTS_DICT
from model.layer.layer import BiasType, ActivationType

def get_basenet_config(dataset_type: int
                       ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    '''
    Generate a model mode dictionary for KAN model.

    Params should be specified only:
        - task_type (categorical)
        - model_type (categorical)
        - dataset_type (categorical)
        - input_size (regressive)
        - output_size (regressive)
    
    Params should be either specified or sampled:
        - hidden_num (minmax)
        - linear_size (minmax)
        - grid_size (minmax)
        - spline_order (minmax)
        - bias_type (multinomial)
        - activation_type (multinomial)
        - activation_param (minmax)
    '''
    param_specified: Dict[str, float] = {
        'task_type': float(TaskType.FORMULA_REGRESSION.value),
        'model_type': float(ModelType.KAN.value),
        'dataset_type': float(dataset_type),
        'input_size': float(FEATURES_DICT[dataset_type]),
        'output_size': float(OUTPUTS_DICT[dataset_type]),
        'hidden_num': float(1),
        'linear_size': float(5),
        'grid_size': float(10 if dataset_type in 
            [DatasetType.SPECIAL_JV.value, DatasetType.SPECIAL_YV.value] else 5),
        'spline_order': float(3),
        'bias_type': float(BiasType.WITH_BIAS.value),
        'activation_type': float(ActivationType.SILU.value),
        'activation_param': float(0.0),
    }

    # For minmax, define as list of two values [min, max], while for
    #   multinomial, define as list of probabilities for each type.
    #   Empty under this case given that all params are specified.
    param_sampled: Dict[str, List[float]] = {}
    return param_specified, param_sampled

def get_basenet_test_config(dataset_type: int
                            ) -> List[Dict[str, float]]:
    '''
    Get the configuration for the base network training.
    '''
    # The i-th value of the list is the i-th test config
    test_params_dict: Dict[str, List[float]] = {
        'task_type': [float(TaskType.FORMULA_REGRESSION.value)],
        'model_type': [float(ModelType.KAN.value)],
        'dataset_type': [float(dataset_type)],
        'input_size': [float(FEATURES_DICT[dataset_type])],
        'output_size': [float(OUTPUTS_DICT[dataset_type])],
        'hidden_num': [float(1)],
        'linear_size': [float(5)],
        'grid_size': [float(10 if dataset_type in 
            [DatasetType.SPECIAL_JV.value, DatasetType.SPECIAL_YV.value] else 5)],
        'spline_order': [float(3)],
        'bias_type': [float(BiasType.WITH_BIAS.value)],
        'activation_type': [float(ActivationType.SILU.value)],
        'activation_param': [float(0.0)],
    }
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
