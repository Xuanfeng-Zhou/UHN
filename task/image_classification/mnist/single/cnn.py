'''
Configuration file for base network configurations
'''
from typing import Dict, List, Tuple
from model.model.model import TaskType, ModelType, DatasetType, FEATURES_DICT, OUTPUTS_DICT
from model.layer.layer import StageWisePoolingType, InputPoolingReshapeType, \
    BiasType, NormType, ActivationType, ShortcutType

def get_basenet_config() -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    '''
    Get the configuration for the base network training.

    Params should be specified only:
        - task_type (categorical)
        - model_type (categorical)
        - dataset_type (categorical)
        - input_channel_dim (regressive)
        - input_size (regressive)
        - output_size (regressive)

    Params should be either specified or sampled:
        Convolutional part:
            - cnn_stage_num (minmax)
            - cnn_layer_num_per_stage (minmax)
            - stage_wise_pooling_type (multinomial)
            - conv_channel_dim (minmax)
            - kernel_size (minmax)
            - group_num (minmax)
            - final_pooling_type (multinomial)
        MLP part:
            - hidden_num (minmax)
            - linear_size (minmax)
        Common:
            - bias_type (multinomial)
            - norm_type (multinomial)
            - activation_type (multinomial)
            - activation_param (minmax)
            - dropout_rate (minmax)
            - shortcut_type (multinomial)
    '''
    param_specified: Dict[str, float] = {
        'task_type': float(TaskType.IMAGE_CLASSIFICATION.value),
        'model_type': float(ModelType.CNN.value),
        'dataset_type': float(DatasetType.MNIST.value),
        'input_channel_dim': float(1),
        'input_size': float(FEATURES_DICT[DatasetType.MNIST.value]),
        'output_size': float(OUTPUTS_DICT[DatasetType.MNIST.value]),
        # Convolutional part
        'cnn_stage_num': float(4),
        'cnn_layer_num_per_stage': float(2),
        'stage_wise_pooling_type': float(StageWisePoolingType.CONV_POOLING.value),
        'conv_channel_dim': float(16),
        'kernel_size': float(3),
        'group_num': float(4),
        'final_pooling_type': float(InputPoolingReshapeType.AVG_POOLING.value),
        # MLP part
        'hidden_num': float(0),
        'linear_size': float(0),
        # Common
        'bias_type': float(BiasType.WITH_BIAS.value),
        'norm_type': float(NormType.ACTIVATION_NORM.value),
        'activation_type': float(ActivationType.LEAKY_RELU.value),
        'activation_param': float(0.1),
        'dropout_rate': float(0.0),
        'shortcut_type': float(ShortcutType.STRAIGHT.value),
    }

    # For minmax, define as list of two values [min, max], while for
    #   multinomial, define as list of probabilities for each type.
    #   Empty under this case given that all params are specified.
    param_sampled: Dict[str, List[float]] = {}
    return param_specified, param_sampled

def get_basenet_test_config() -> List[Dict[str, float]]:
    '''
    Get the configuration for the base network training.
    '''
    # The i-th value of the list is the i-th test config
    test_params_dict: Dict[str, List[float]] = {
        'task_type': [float(TaskType.IMAGE_CLASSIFICATION.value)],
        'model_type': [float(ModelType.CNN.value)],
        'dataset_type': [float(DatasetType.MNIST.value)],
        'input_channel_dim': [float(1)],
        'input_size': [float(FEATURES_DICT[DatasetType.MNIST.value])],
        'output_size': [float(OUTPUTS_DICT[DatasetType.MNIST.value])],
        # Convolutional part
        'cnn_stage_num': [float(4)],
        'cnn_layer_num_per_stage': [float(2)],
        'stage_wise_pooling_type': [float(StageWisePoolingType.CONV_POOLING.value)],
        'conv_channel_dim': [float(16)],
        'kernel_size': [float(3)],
        'group_num': [float(4)],
        'final_pooling_type': [float(InputPoolingReshapeType.AVG_POOLING.value)],
        # MLP part
        'hidden_num': [float(0)],
        'linear_size': [float(0)],
        # Common
        'bias_type': [float(BiasType.WITH_BIAS.value)],
        'norm_type': [float(NormType.ACTIVATION_NORM.value)],
        'activation_type': [float(ActivationType.LEAKY_RELU.value)],
        'activation_param': [float(0.1)],
        'dropout_rate': [float(0.0)],
        'shortcut_type': [float(ShortcutType.STRAIGHT.value)],
    }
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
