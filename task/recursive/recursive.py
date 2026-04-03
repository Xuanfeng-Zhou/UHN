'''
Configuration file for base network configurations
'''
from typing import Dict, List, Tuple
from model.model.model import TaskType, ModelType
from model.layer.layer import BiasType, ActivationType, NormType, ShortcutType

def get_basenet_config(dataset_type: int
                       ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    '''
    Generate a model mode dictionary for Recursive model.

    Params should be specified only:
        - task_type (categorical)
        - model_type (categorical)
        - dataset_type (categorical)
        - num_structure_freqs (regressive)
        - num_index_freqs (regressive)
    
    Params should be either specified or sampled:
        Multihead attention part:
            - num_heads (minmax)
        MLP part:
            - hidden_num (minmax)
            - linear_size (minmax)
        Common:
            - bias_type (multinomial)
            - norm_type (multinomial)
            - shortcut_type (multinomial)
            - activation_type (multinomial)
            - activation_param (minmax)
            - dropout_rate (minmax)
    '''
    param_specified: Dict[str, float] = {
        'task_type': float(TaskType.RECURSIVE.value),
        'model_type': float(ModelType.RECURSIVE.value),
        'dataset_type': float(dataset_type),
        'num_structure_freqs': float(32),
        'num_index_freqs': float(1024),
        'num_heads': float(4),
        # The hidden num should be at least 1 to be compatible with structural encoder
        'hidden_num': float(5),
        'linear_size': float(64),
        'bias_type': float(BiasType.WITH_BIAS.value),
        'norm_type': float(NormType.ACTIVATION_NORM.value),
        'shortcut_type': float(ShortcutType.STRAIGHT.value),
        'activation_type': float(ActivationType.LEAKY_RELU.value),
        'activation_param': float(0.1),
        'dropout_rate': float(0.0),
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
        'task_type': [float(TaskType.RECURSIVE.value)],
        'model_type': [float(ModelType.RECURSIVE.value)],
        'dataset_type': [float(dataset_type)],
        'num_structure_freqs': [float(32)],
        'num_index_freqs': [float(1024)],
        'num_heads': [float(4)],
        # The hidden num should be at least 1 to be compatible with structural encoder
        'hidden_num': [float(5)],
        'linear_size': [float(64)],
        'bias_type': [float(BiasType.WITH_BIAS.value)],
        'norm_type': [float(NormType.ACTIVATION_NORM.value)],
        'shortcut_type': [float(ShortcutType.STRAIGHT.value)],
        'activation_type': [float(ActivationType.LEAKY_RELU.value)],
        'activation_param': [float(0.1)],
        'dropout_rate': [float(0.0)],
    }
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
