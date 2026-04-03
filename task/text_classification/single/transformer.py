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
        # Multihead attention part
        'encoder_num': float(2
            if dataset_type == DatasetType.AG_NEWS.value else 1),
        'num_heads': float(2),
        # MLP part
        'linear_layer_num_per_encoder': float(2),
        # Common
        'embedding_dim': float(64),
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
        'task_type': [float(TaskType.TEXT_CLASSIFICATION.value)],
        'model_type': [float(ModelType.TRANSFORMER.value)],
        'dataset_type': [float(dataset_type)],
        'vocab_size': [float(5000)],
        'output_size': [float(OUTPUTS_DICT[dataset_type])],
        'max_sequence_length': [float(128 
            if dataset_type == DatasetType.AG_NEWS.value else 512)],
        # Multihead attention part
        'encoder_num': [float(2
            if dataset_type == DatasetType.AG_NEWS.value else 1)],
        'num_heads': [float(2)],
        # MLP part
        'linear_layer_num_per_encoder': [float(2)],
        # Common
        'embedding_dim': [float(64)],
        'bias_type': [float(BiasType.WITH_BIAS.value)],
        'shortcut_type': [float(ShortcutType.STRAIGHT.value)],
        'norm_type': [float(NormType.ACTIVATION_NORM.value)],
        'activation_type': [float(ActivationType.LEAKY_RELU.value)],
        'activation_param': [float(0.1)],
        'dropout_rate': [float(0.2
            if dataset_type == DatasetType.AG_NEWS.value else 0.4)],
    }
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
