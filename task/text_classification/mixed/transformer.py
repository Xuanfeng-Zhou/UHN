'''
Configuration file for base network configurations
'''
from typing import Dict, List, Tuple
from model.model.model import TaskType, ModelType, OUTPUTS_DICT, DatasetType
from model.layer.layer import BiasType, NormType, ActivationType, ShortcutType

def get_basenet_config(dataset_type: int
                       ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    '''
    Generate a model mode dictionary for Transformer model.

    Params should be specified only:
        - task_type (categorical)
        - model_type (categorical)
        - dataset_type (categorical)
        - vocab_size (regressive)
        - output_size (regressive)
        - max_sequence_length (regressive)
        
    Params should be either specified or sampled:
        Multihead attention part:
            - encoder_num (minmax)
            - num_heads (minmax)
        MLP part:
            - linear_layer_num_per_encoder (minmax)
        Common:
            - embedding_dim (minmax)
            - bias_type (multinomial)
            - shortcut_type (multinomial)
            - norm_type (multinomial)
            - activation_type (multinomial)
            - activation_param (minmax)
            - dropout_rate (minmax)
    '''
    param_specified: Dict[str, float] = {
        'task_type': float(TaskType.TEXT_CLASSIFICATION.value),
        'model_type': float(ModelType.TRANSFORMER.value),
        'dataset_type': float(dataset_type),
        'vocab_size': float(5000),
        'output_size': float(OUTPUTS_DICT[dataset_type]),
        'max_sequence_length': float(128
            if dataset_type == DatasetType.AG_NEWS.value else 512),
        # Common
        'bias_type': float(BiasType.WITH_BIAS.value),
        'shortcut_type': float(ShortcutType.STRAIGHT.value),
        'norm_type': float(NormType.ACTIVATION_NORM.value),
        'activation_type': float(ActivationType.LEAKY_RELU.value),
        'activation_param': float(0.1),
        'dropout_rate': float(0.2
            if dataset_type == DatasetType.AG_NEWS.value else 0.4),        
    }

    # For minmax, define as list of two values [min, max], while for
    #   multinomial, define as list of probabilities for each type.
    param_sampled: Dict[str, List[float]] = {
        # Multihead attention part
        'encoder_num_range': [
            float(1), # min
            float(4), # max
            ],
        'num_heads_range': [
            float(1), # min
            float(8), # max
            ],
        # MLP part
        'linear_layer_num_per_encoder_range': [
            float(1), # min
            float(3), # max
        ],
        # Common
        'embedding_dim_range': [
            float(32), # min
            float(128), # max
        ]
    }
    return param_specified, param_sampled

def get_basenet_test_config(dataset_type: int
                            ) -> List[Dict[str, float]]:
    '''
    Get the configuration for the base network training.
    '''
    # The i-th value of the list is the i-th test config
    test_params_dict: Dict[str, List[float]] = {
        'task_type': [float(TaskType.TEXT_CLASSIFICATION.value)] * 8,
        'model_type': [float(ModelType.TRANSFORMER.value)] * 8,
        'dataset_type': [float(dataset_type)] * 8,
        'vocab_size': [float(5000)] * 8,
        'output_size': [float(OUTPUTS_DICT[dataset_type])] * 8,
        'max_sequence_length': [float(128
            if dataset_type == DatasetType.AG_NEWS.value else 512)] * 8,
        # Common
        'bias_type': [float(BiasType.WITH_BIAS.value)] * 8,
        'shortcut_type': [float(ShortcutType.STRAIGHT.value)] * 8,
        'norm_type': [float(NormType.ACTIVATION_NORM.value)] * 8,
        'activation_type': [float(ActivationType.LEAKY_RELU.value)] * 8,
        'activation_param': [float(0.1)] * 8,
        'dropout_rate': [float(0.2
            if dataset_type == DatasetType.AG_NEWS.value else 0.4)] * 8,

        # Variations of the base network, 6 in-range test configs including baseline 
        #   and 2 out-of-range test configs
        # Multihead attention part
        "encoder_num":                    [ 1.0,  1.0,  2.0,  2.0,  3.0,   2.0,   3.0,  3.0],
        "num_heads":                      [ 1.0,  2.0,  2.0,  3.0,  3.0,   4.0,   4.0,  2.0],
        # MLP part
        "linear_layer_num_per_encoder":   [ 1.0,  2.0,  2.0,  3.0,  3.0,   2.0,   3.0,  3.0],
        # Common
        "embedding_dim":                  [32.0, 64.0, 64.0, 96.0, 96.0, 128.0, 128.0, 64.0],
        # Specify the linear layer num for each encoder, respectively
        'linear_layer_num_per_encoder_0': [ 1.0,  2.0,  2.0,  3.0,  2.0,   1.0,   3.0,  1.0],
        'linear_layer_num_per_encoder_1': [ 0.0,  0.0,  2.0,  1.0,  1.0,   2.0,   3.0,  3.0],
        'linear_layer_num_per_encoder_2': [ 0.0,  0.0,  0.0,  0.0,  3.0,   0.0,   3.0,  2.0]
    }
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
