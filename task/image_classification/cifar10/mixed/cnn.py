'''
Configuration file for base network configurations
'''
from typing import Dict, List, Tuple
from model.model.model import TaskType, ModelType, DatasetType, FEATURES_DICT, OUTPUTS_DICT
from model.layer.layer import StageWisePoolingType, InputPoolingReshapeType, \
    BiasType, NormType, ActivationType, ShortcutType

def get_basenet_config(mix_type: str='depth_and_width') -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    '''
    Get the configuration for the base network training.
    Args:
        mix_type: mixing type of the model, "depth" or "width" or "depth_and_width"

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
        'dataset_type': float(DatasetType.CIFAR10.value),
        'input_channel_dim': float(3),
        'input_size': float(FEATURES_DICT[DatasetType.CIFAR10.value]),
        'output_size': float(OUTPUTS_DICT[DatasetType.CIFAR10.value]),
        # Convolutional part
        'cnn_stage_num': float(4),
        'stage_wise_pooling_type': float(StageWisePoolingType.CONV_POOLING.value),
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
        'shortcut_type': float(ShortcutType.STRAIGHT.value),
        'dropout_rate': float(0.0),
        'activation_param': float(0.1),
    }

    # For minmax, define as list of two values [min, max], while for
    #   multinomial, define as list of probabilities for each type.
    if mix_type == 'depth':
        param_sampled: Dict[str, List[float]] = {
            # Convolutional part
            'cnn_layer_num_per_stage_range': [
                float(6), # min
                float(10)  # max
            ],                      
            'conv_channel_dim_range': [
                float(16), # min
                float(16)  # max
            ],        
        }
    elif mix_type == 'width':
        param_sampled: Dict[str, List[float]] = {
            # Convolutional part
            'cnn_layer_num_per_stage_range': [
                float(6), # min
                float(6)  # max
            ],         
            'conv_channel_dim_range': [
                float(16), # min
                float(32)  # max
            ],
        }
    elif mix_type == 'depth_and_width':
        param_sampled: Dict[str, List[float]] = {
            # Convolutional part
            'cnn_layer_num_per_stage_range': [
                float(6), # min
                float(8)  # max                
            ],
            'conv_channel_dim_range': [
                float(16), # min
                float(32), # max          
            ],
        }
    else:
        raise ValueError(f"Invalid sweep mode: {mix_type}. "
                         f"Expected 'depth' or 'width'.")
    return param_specified, param_sampled

def get_basenet_test_config(mix_type: str='depth_and_width') -> List[Dict[str, float]]:
    '''
    Get the configuration for the base network training.
    Args:
        mix_type: mixing type of the model, "depth" or "width" or "depth_and_width"
    '''
    # The i-th value of the list is the i-th test config
    if mix_type == 'depth':
        test_params_dict: Dict[str, List[float]] = {
            'task_type': [float(TaskType.IMAGE_CLASSIFICATION.value)] * 8,
            'model_type': [float(ModelType.CNN.value)] * 8,
            'dataset_type': [float(DatasetType.CIFAR10.value)] * 8,
            'input_channel_dim': [float(3)] * 8,
            'input_size': [float(FEATURES_DICT[DatasetType.CIFAR10.value])] * 8,
            'output_size': [float(OUTPUTS_DICT[DatasetType.CIFAR10.value])] * 8,
            # Convolutional part
            'stage_wise_pooling_type': [float(StageWisePoolingType.CONV_POOLING.value)] * 8,
            'kernel_size': [float(3)] * 8,
            'group_num': [float(4)] * 8,
            'final_pooling_type': [float(InputPoolingReshapeType.AVG_POOLING.value)] * 8,
            # MLP part
            'hidden_num': [float(0)] * 8,
            'linear_size': [float(0)] * 8,
            # Common
            'bias_type': [float(BiasType.WITH_BIAS.value)] * 8,
            'norm_type': [float(NormType.ACTIVATION_NORM.value)] * 8,
            'activation_type': [float(ActivationType.LEAKY_RELU.value)] * 8,
            'shortcut_type': [float(ShortcutType.STRAIGHT.value)] * 8,   
            'activation_param': [0.10] * 8,
            'dropout_rate': [0.00] * 8,
            # Variations of the base network, 6 in-range test configs including baseline 
            #   and 2 out-of-range test configs             
            # Convolutional part
            'cnn_stage_num':            [ 4.0] * 8,
            'cnn_layer_num_per_stage':  [ 6.0,  10.0,  14.0,  18.0, 14.0, 18.0, 15.0, 16.0],
            'conv_channel_dim':         [16.0] * 8,        
            # Specify the layer number for each stage, respectively
            # 2nd stage
            'cnn_layer_num_per_stage_1':  [ 6.0, 10.0, 14.0, 18.0,  6.0, 12.0, 15.0, 13.0],
            # 3rd stage
            'cnn_layer_num_per_stage_2':  [ 6.0, 10.0, 14.0, 18.0, 10.0, 18.0,  8.0, 10.0],
            # 4th stage
            'cnn_layer_num_per_stage_3':  [ 6.0, 10.0, 14.0, 18.0, 14.0,  9.0, 12.0, 16.0],
        }            
    elif mix_type == 'width':
        test_params_dict: Dict[str, List[float]] = {
            'task_type': [float(TaskType.IMAGE_CLASSIFICATION.value)] * 8,
            'model_type': [float(ModelType.CNN.value)] * 8,
            'dataset_type': [float(DatasetType.CIFAR10.value)] * 8,
            'input_channel_dim': [float(3)] * 8,
            'input_size': [float(FEATURES_DICT[DatasetType.CIFAR10.value])] * 8,
            'output_size': [float(OUTPUTS_DICT[DatasetType.CIFAR10.value])] * 8,
            # Convolutional part
            'stage_wise_pooling_type': [float(StageWisePoolingType.CONV_POOLING.value)] * 8,
            'kernel_size': [float(3)] * 8,
            'group_num': [float(4)] * 8,
            'final_pooling_type': [float(InputPoolingReshapeType.AVG_POOLING.value)] * 8,
            # MLP part
            'hidden_num': [float(0)] * 8,
            'linear_size': [float(0)] * 8,
            # Common
            'bias_type': [float(BiasType.WITH_BIAS.value)] * 8,
            'norm_type': [float(NormType.ACTIVATION_NORM.value)] * 8,
            'activation_type': [float(ActivationType.LEAKY_RELU.value)] * 8,
            'shortcut_type': [float(ShortcutType.STRAIGHT.value)] * 8,   
            'activation_param': [0.10] * 8,
            'dropout_rate': [0.00] * 8,
            # Variations of the base network, 6 in-range test configs including baseline 
            #   and 2 out-of-range test configs             
            # Convolutional part
            'cnn_stage_num':            [ 4.0] * 8,
            'cnn_layer_num_per_stage':  [ 6.0] * 8,
            'conv_channel_dim':         [16.0, 20.0, 24.0,  28.0,  32.0,  20.0, 24.0, 28.0],        
            # Specify the dimension for each stage, respectively
            # 3rd stage, at layer idx 7
            'conv_channel_dim_7':       [32.0, 40.0, 48.0,  56.0,  64.0,  56.0, 52.0, 44.0],
            # 4th stage, at layer idx 13
            'conv_channel_dim_13':      [64.0, 80.0, 96.0, 112.0, 128.0, 128.0, 80.0, 96.0],
        }
    else:
        raise ValueError(f"Invalid sweep mode: {mix_type}. "
                         f"Expected 'depth' or 'width'.")
    
    # Create a list of dictionaries for each test config
    test_config_list: List[Dict[str, float]] = [
        dict(zip(test_params_dict.keys(), values)) 
        for values in zip(*test_params_dict.values())
    ]
    return test_config_list
