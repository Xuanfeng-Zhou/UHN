'''
Configuration file for base network configurations
'''
from typing import Dict, List, Tuple
from model.model.model import TaskType, ModelType, DatasetType, FEATURES_DICT, OUTPUTS_DICT
from model.layer.layer import BiasType, ActivationType

def get_basenet_config(dataset_type: int
                       ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    '''
    Generate model and layers mode dictionaries for GAT model.

    Params should be specified only:
        - task_type (categorical)
        - model_type (categorical)
        - dataset_type (categorical)
        - input_size (regressive)
        - output_size (regressive)
    
    Params should be either specified or sampled:
        - hidden_num (minmax)
        - num_heads_hidden (minmax)
        - linear_size_per_head (minmax)
        - num_heads_output (minmax)
        - bias_type (multinomial)
        - activation_type (multinomial)
        - activation_param (minmax)
        - dropout_rate (minmax)
    '''
    param_specified: Dict[str, float] = {
        'task_type': float(TaskType.GRAPH_NODE_CLASSIFICATION.value),
        'model_type': float(ModelType.GAT.value),
        'dataset_type': float(dataset_type),
        'input_size': float(FEATURES_DICT[dataset_type]),
        'output_size': float(OUTPUTS_DICT[dataset_type]),
        'num_heads_hidden': float(8),
        'num_heads_output': float(8 if dataset_type == DatasetType.PUBMED.value else 1),
        'bias_type': float(BiasType.WITH_BIAS.value),
        'activation_type': float(ActivationType.ELU.value),
        'activation_param': float(1.0),
        'dropout_rate': float(0.6),
    }

    # For minmax, define as list of two values [min, max], while for
    #   multinomial, define as list of probabilities for each type.
    param_sampled: Dict[str, List[float]] = {
        'hidden_num_range': [
            float(1), # min
            float(2), # max
            ],
        'linear_size_per_head_range': [
            float(4), # min
            float(16), # max
            ],
    }
    return param_specified, param_sampled

def get_basenet_test_config(dataset_type: int
                            ) -> List[Dict[str, float]]:
    '''
    Get the configuration for the base network training.
    '''
    # The i-th value of the list is the i-th test config
    test_params_dict: Dict[str, List[float]] = {
        'task_type': [float(TaskType.GRAPH_NODE_CLASSIFICATION.value)] * 2,
        'model_type': [float(ModelType.GAT.value)] * 2,
        'dataset_type': [float(dataset_type)] * 2,
        'input_size': [float(FEATURES_DICT[dataset_type])] * 2,
        'output_size': [float(OUTPUTS_DICT[dataset_type])] * 2,
        'hidden_num': [float(1)] * 2,
        'num_heads_hidden': [float(8)] * 2,
        'linear_size_per_head': [float(8),
                                 float(16)],
        'num_heads_output': [float(8 if dataset_type == DatasetType.PUBMED.value else 1)] * 2,
        'bias_type': [float(BiasType.WITH_BIAS.value)] * 2,
        'activation_type': [float(ActivationType.ELU.value)] * 2,
        'activation_param': [float(1.0)] * 2,
        'dropout_rate': [float(0.6)] * 2,
    }
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
