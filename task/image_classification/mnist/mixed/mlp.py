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
        'cnn_stage_num': float(0),
        'cnn_layer_num_per_stage': float(0),
        'stage_wise_pooling_type': float(StageWisePoolingType.NONE.value),
        'conv_channel_dim': float(0),
        'kernel_size': float(0),
        'group_num': float(0),
        'final_pooling_type': float(InputPoolingReshapeType.RESHAPE_TO_2D.value),
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
    param_sampled: Dict[str, List[float]] = {
        # MLP part
        'hidden_num_range': [
            float(2), # min
            float(3)  # max
            ],
        'linear_size_range': [
            float(64), # min
            float(256) # max
            ],
    }
    return param_specified, param_sampled

def get_basenet_test_config() -> List[Dict[str, float]]:
    '''
    Get the configuration for the base network training.
    '''
    # The i-th value of the list is the i-th test config
    test_params_dict: Dict[str, List[float]] = {
        'task_type': [float(TaskType.IMAGE_CLASSIFICATION.value)] * 2,
        'model_type': [float(ModelType.CNN.value)] * 2,
        'dataset_type': [float(DatasetType.MNIST.value)] * 2,
        'input_channel_dim': [float(1)] * 2,
        'input_size': [float(FEATURES_DICT[DatasetType.MNIST.value])] * 2,
        'output_size': [float(OUTPUTS_DICT[DatasetType.MNIST.value])] * 2,
        # Convolutional part
        'cnn_stage_num': [float(0)] * 2,
        'cnn_layer_num_per_stage': [float(0)] * 2,
        'stage_wise_pooling_type': [float(StageWisePoolingType.NONE.value)] * 2,
        'conv_channel_dim': [float(0)] * 2,
        'kernel_size': [float(0)] * 2,
        'group_num': [float(0)] * 2,
        'final_pooling_type': [float(InputPoolingReshapeType.RESHAPE_TO_2D.value)] * 2,
        # MLP part
        'hidden_num': [float(2)] * 2,
        'linear_size': [float(128), 
                        float(256)],
        # Common
        'bias_type': [float(BiasType.WITH_BIAS.value)] * 2,
        'norm_type': [float(NormType.ACTIVATION_NORM.value)] * 2,
        'activation_type': [float(ActivationType.LEAKY_RELU.value)] * 2,
        'activation_param': [float(0.1)] * 2,
        'dropout_rate': [float(0.0)] * 2,
        'shortcut_type': [float(ShortcutType.STRAIGHT.value)] * 2,
    }
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
