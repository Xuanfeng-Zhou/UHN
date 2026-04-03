'''
CNN model encoding & forwarding and modularization
'''
import torch
from .model import ModelUtils, ModelType
from ..layer import LinearLayerUtils, ConvLayerUtils
from ..layer.layer import LayerType, ShortcutType, StageWisePoolingType, \
    ActivationType, NormType, InputPoolingReshapeType, BiasType, ParamType, InitializationType
from typing import List, Dict, Tuple
from optimization import jit_script

@jit_script
def _get_model_mode(task_type: int,
                     dataset_type: int,
                     num_layers: int,
                     cnn_stage_num: int,
                     ) -> Dict[str, float]:
    '''
    Get a model mode dictionary for CNN model.
    '''
    model_mode: Dict[str, float] = ModelUtils.get_model_mode()
    model_mode.update({
        "model_type": float(ModelType.CNN.value),
        "task_type": float(task_type),
        "dataset_type": float(dataset_type),
        "num_layers": float(num_layers),
        "cnn_stage_num": float(cnn_stage_num)
    })
    return model_mode

@jit_script
def _generate_mode(param_specified: Dict[str, float],
                  param_sampled: Dict[str, List[float]]
                  ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    '''
    Generate model and layers structure for CNN model.
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
    global_mode, local_mode = ModelUtils.generate_mode(param_specified, param_sampled)
    
    # --------------------- Generate global mode ---------------------
    # Hidden layers num
    hidden_num: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "minmax")
    # CNN stage num
    cnn_stage_num: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                  "cnn_stage_num", "minmax")
    # Layer num (cnns + hiddens + output)
    # First stage has 1 layer, other stages have cnn_layer_num_per_stage layers
    # Stage transition layer idx
    stage_transition_layer_idx: List[int] = []
    if cnn_stage_num == 0:
        num_conv_layers: int = 0
    else:
        # Sample the number of layers for each stage
        for stage_idx in range(cnn_stage_num):
            if stage_idx == 0:
                stage_layer_num: int = 1
                # Add stage transition layer idx
                stage_transition_layer_idx.append(stage_layer_num)
            else:
                stage_layer_num: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "cnn_layer_num_per_stage", "minmax",
                                                check_idx=True, attribute_idx=stage_idx)
                # Add stage transition layer idx
                stage_transition_layer_idx.append(stage_layer_num + stage_transition_layer_idx[stage_idx - 1])
        # Calculate the number of convolutional layers
        num_conv_layers: int = stage_transition_layer_idx[-1]
    num_linear_layers: int = hidden_num + 1
    num_layers: int = num_conv_layers + num_linear_layers
    # Generate the model mode
    global_mode.update(_get_model_mode(
        task_type=int(param_specified["task_type"]),
        dataset_type=int(param_specified["dataset_type"]),
        num_layers=num_layers,
        cnn_stage_num=cnn_stage_num
    ))
    # --------------------- Generate local mode ---------------------
    conv_channel_factor: int = 1
    input_channel_dim: int = int(param_specified["input_channel_dim"])
    input_size: int = int(param_specified["input_size"])
    # First layer always uses no normalization
    norm_type: int = NormType.NONE.value
    group_num: int = 0
    for layer_idx in range(num_layers):
        # Bias type and dropout rate
        if layer_idx == num_layers - 1:
            # Output layer always uses bias and no dropout
            bias_type: int = BiasType.WITH_BIAS.value
            dropout_rate: float = 0.0
        else:
            bias_type: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                                    "bias_type", "multinomial")
            dropout_rate: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                                "dropout_rate", "minmax")                
        # Sample the norm type for the next layer
        if layer_idx + 1 == num_conv_layers:
            # First layer after conv uses no normalization
            next_norm_type: int = NormType.NONE.value
        else:
            next_norm_type: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                    "norm_type", "multinomial")
        # Activation type and activation param
        if layer_idx == 0 or layer_idx == num_conv_layers:
            # First layer or first layer after conv uses no activation
            activation_type: int = ActivationType.NONE.value
            activation_param: float = 0.0
        else:
            activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                        "activation_type", "multinomial")
            activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                            "activation_param", "minmax")
        if layer_idx < num_conv_layers:
            # Sample the group num for next layer
            if next_norm_type == NormType.ACTIVATION_NORM.value:
                next_group_num: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                        "group_num", "minmax",
                                                        check_idx=True, attribute_idx=layer_idx + 1)
            else:
                next_group_num: int = 0

            # Kernel size, use odd kernel size only
            kernel_size: int = ModelUtils.sample_odd_int_param(param_specified, param_sampled, 
                                                                    "kernel_size")
            # Times multiplier by 2 for the channel size after each stage (except the first stage)
            if (layer_idx > 1) and (layer_idx in stage_transition_layer_idx):
                conv_channel_factor *= 2
                # Add stage wise pooling type
                stage_wise_pooling_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                                        "stage_wise_pooling_type", "multinomial")
                # Use no shortcut type for layer with stage wise pooling
                shortcut_type: int = ShortcutType.NONE.value
            else:
                # No stage wise pooling type otherwise
                stage_wise_pooling_type: int = StageWisePoolingType.NONE.value
                if layer_idx == 0:
                    # Use no shortcut type for the first layer
                    shortcut_type: int = ShortcutType.NONE.value
                else:
                    # Sample the shortcut type
                    shortcut_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                                "shortcut_type", "multinomial")
            # Output channel dim
            if shortcut_type == ShortcutType.NONE.value:
                output_channel_dim: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "conv_channel_dim", "minmax",
                                                multiplier=conv_channel_factor, divisable_by=next_group_num,
                                                check_idx=True, attribute_idx=layer_idx)
            else:
                # Identical dimension for shortcut
                output_channel_dim: int = input_channel_dim    
                # Check if the group num is valid
                if next_norm_type == NormType.ACTIVATION_NORM.value:
                    # If next layer uses activation norm, group num should be a divisor of the output channel dim
                    if output_channel_dim % next_group_num != 0:
                        # Resample the group num if it is not a divisor of the output channel dim
                        sampled_value, _ = ModelUtils.sample_int_param_no_raise(
                            param_specified, param_sampled,
                            "group_num", "minmax",
                            divisor_of=output_channel_dim,
                            check_idx=True, attribute_idx=layer_idx + 1
                        )
                        if sampled_value == -1:
                            # If resampling failed, use 1 as the group num
                            next_group_num = 1
                        else:
                            # Use the sampled value as the group num
                            next_group_num = sampled_value
            # Generate the layer mode for convolutional layer
            layer_mode: Dict[str, float] = ConvLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                bias_type=bias_type,
                norm_type=norm_type,
                shortcut_type=shortcut_type,
                output_channel_dim=output_channel_dim,
                input_channel_dim=input_channel_dim,
                activation_type=activation_type,
                activation_param=activation_param,
                dropout_rate=dropout_rate,
                group_num=group_num,
                kernel_size=kernel_size,
                stage_wise_pooling_type=stage_wise_pooling_type,
                initialization_type=InitializationType.DEFAULT.value
            )
            # Update the input channel dim for the next layer
            input_channel_dim = output_channel_dim
            # Update the group num for the next layer
            group_num = next_group_num
        else:
            # Shortcut type
            if layer_idx == 0 or layer_idx == num_conv_layers or layer_idx == num_layers - 1:
                # First, right after conv or last layer uses no shortcut
                shortcut_type: int = ShortcutType.NONE.value
            else:
                shortcut_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                            "shortcut_type", "multinomial")
            # Input pooling or reshape type
            if layer_idx == 0:
                # First layer uses reshape
                input_pooling_reshape_type: int = InputPoolingReshapeType.RESHAPE_TO_2D.value
            elif layer_idx == num_conv_layers:
                # Sample pooling method after the last conv layer (avg or max pooling only)
                input_pooling_reshape_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                                        "final_pooling_type", "multinomial")
                input_size = input_channel_dim
            else:
                # Use no pooling or reshape for other layers
                input_pooling_reshape_type: int = InputPoolingReshapeType.NONE.value
            # Output size
            if layer_idx == num_layers - 1:
                output_size: int = int(param_specified["output_size"])
            else:
                if shortcut_type == ShortcutType.NONE.value:
                    output_size: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "linear_size", "minmax")
                else:
                    output_size: int = input_size
            # Generate the layer mode for linear layer
            layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                bias_type=bias_type,
                norm_type=norm_type,
                shortcut_type=shortcut_type,
                output_size=output_size,
                input_size=input_size,
                activation_type=activation_type,
                activation_param=activation_param,
                dropout_rate=dropout_rate,
                input_pooling_reshape_type=input_pooling_reshape_type,
                initialization_type=InitializationType.DEFAULT.value
            )
            # Update the input size for the next layer
            input_size = output_size
        # Append the layer mode
        local_mode.append(layer_mode)
        # Update the normalization type for the next layer
        norm_type = next_norm_type
    return global_mode, local_mode

@jit_script
def _generate_mode_for_size(param_specified: Dict[str, float]
                           ) -> List[Dict[str, float]]:
    '''
    Generate a slim version of structures for CNN model, only for model size calculation.
    Params should be specified:
        Common:
            - input_channel_dim
            - input_size
            - output_size
        Convolution part:
            - cnn_stage_num
            - cnn_layer_num_per_stage
            - conv_channel_dim
            - kernel_size
        MLP part:
            - hidden_num
            - linear_size
    '''
    local_mode = ModelUtils.generate_mode_for_size(param_specified)
    # Several global params
    hidden_num: int = int(param_specified["hidden_num"])
    cnn_stage_num: int = int(param_specified["cnn_stage_num"])
    cnn_layer_num_per_stage: int = int(param_specified["cnn_layer_num_per_stage"])
    # Layer num (cnns + hiddens + output)
    # First stage has 1 layer, other stages have cnn_layer_num_per_stage layers
    if cnn_stage_num == 0:
        num_conv_layers: int = 0
    else:
        num_conv_layers: int = (cnn_stage_num - 1) * cnn_layer_num_per_stage + 1
    num_linear_layers: int = hidden_num + 1
    num_layers: int = num_conv_layers + num_linear_layers
    # --------------------- Generate local mode ---------------------
    conv_channel_factor: int = 1
    input_channel_dim: int = int(param_specified["input_channel_dim"])
    input_size: int = int(param_specified["input_size"])
    for layer_idx in range(num_layers):
        # Ignore most attributes but leave the out & in channel dim remained
        # Always use bias in this case
        bias_type: int = BiasType.WITH_BIAS.value            
        if layer_idx < num_conv_layers:
            # Convolutional part
            kernel_size: int = int(param_specified["kernel_size"])
            # Times multiplier by 2 for the channel size after each stage (except the first stage)
            if layer_idx > 1 and layer_idx % cnn_layer_num_per_stage == 1:
                conv_channel_factor *= 2                    
            output_channel_dim: int = int(param_specified["conv_channel_dim"]) * conv_channel_factor
            # Generate the layer mode for convolution layer, and only fill size related attributes
            layer_mode: Dict[str, float] = ConvLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                bias_type=bias_type,
                norm_type=NormType.NONE.value,
                shortcut_type=ShortcutType.NONE.value,
                output_channel_dim=output_channel_dim,
                input_channel_dim=input_channel_dim,
                activation_type=ActivationType.NONE.value,
                activation_param=0.0,
                dropout_rate=0.0,
                group_num=0,
                kernel_size=kernel_size,
                stage_wise_pooling_type=StageWisePoolingType.NONE.value,
                initialization_type= InitializationType.DEFAULT.value
            )      
            # Update the input channel dim for the next layer
            input_channel_dim = output_channel_dim
        else:
            # MLP part
            # Input pooling or reshape type
            if layer_idx != 0 and layer_idx == num_conv_layers:
                input_size = input_channel_dim

            if layer_idx == num_layers - 1:
                output_size: int = int(param_specified["output_size"])
            else:
                output_size: int = int(param_specified["linear_size"])
            # Generate the layer mode for linear layer, and only fill size related attributes
            layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                bias_type=bias_type,
                norm_type=NormType.NONE.value,
                shortcut_type=ShortcutType.NONE.value,
                output_size=output_size,
                input_size=input_size,
                activation_type=ActivationType.NONE.value,
                activation_param=0.0,
                dropout_rate=0.0,
                input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                initialization_type=InitializationType.DEFAULT.value
            )
            # Update the input size for the next layer
            input_size = output_size
        # Append the layer mode
        local_mode.append(layer_mode)
    return local_mode

@jit_script
def _retrieve_max_memory_size(param_specified: Dict[str, float],
                             param_sampled: Dict[str, List[float]]
                             ) -> Tuple[int, int, int, int, int]:
    '''
    Retrieve the maximum should-be pre-allocated memory size for encoding,
        shared elements and arange tensor for CNN model.
    '''
    global_structure_size, local_structure_size, encode_memory_size, \
        shared_element_buffer_size, arange_tensor_size = \
        ModelUtils.retrieve_max_memory_size(param_specified, param_sampled)
    # Hidden layers num
    hidden_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "max")        
    # Hidden size
    hidden_size_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "linear_size", "max")
    # CNN stage num
    cnn_stage_num_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                  "cnn_stage_num", "min")
    cnn_stage_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                  "cnn_stage_num", "max")        
    # CNN layer num per stage
    cnn_layer_num_per_stage_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                            "cnn_layer_num_per_stage", "min")
    cnn_layer_num_per_stage_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                            "cnn_layer_num_per_stage", "max")
    # Convolution layer num
    if cnn_stage_num_min == 0:
        num_conv_layers_min: int = 0
    else:
        num_conv_layers_min: int = (cnn_stage_num_min - 1) * cnn_layer_num_per_stage_min + 1
    if cnn_stage_num_max == 0:
        num_conv_layers_max: int = 0
    else:
        num_conv_layers_max: int = (cnn_stage_num_max - 1) * cnn_layer_num_per_stage_max + 1
    # Check if no convolutional layer and just MLP
    if num_conv_layers_min == 0 or num_conv_layers_max == 0:
        # Could be just MLP or only can be just MLP
        param_specified_mlp_only: Dict[str, float] = {
            # Common
            "input_channel_dim": param_specified["input_channel_dim"],
            "input_size": param_specified["input_size"],
            "output_size": param_specified["output_size"],
            # Convolution part
            "cnn_stage_num": 0.0,
            "cnn_layer_num_per_stage": 0.0,
            "conv_channel_dim": 0.0,
            "kernel_size": 0.0,
            # MLP part
            "hidden_num": float(hidden_num_max),
            "linear_size": float(hidden_size_max)
        }
        mode_mlp_only: List[Dict[str, float]] = _generate_mode_for_size(param_specified_mlp_only)
        # Implement the calculation for the maximum memory size and update
        global_structure_size, local_structure_size, encode_memory_size, \
            shared_element_buffer_size, arange_tensor_size = \
            ModelUtils.update_max_memory_size(global_structure_size,
                                           local_structure_size,
                                           encode_memory_size,
                                           shared_element_buffer_size,
                                           arange_tensor_size,
                                           mode_mlp_only)
    # When there are convolutional layers
    if num_conv_layers_max > 0:
        # Convolution dimension
        conv_channel_dim_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "conv_channel_dim", "max")
        # Kernel size
        kernel_size_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "kernel_size", "max")
        param_specified_conv = {
            # Common
            "input_channel_dim": param_specified["input_channel_dim"],
            "input_size": param_specified["input_size"],
            "output_size": param_specified["output_size"],
            # Convolution part
            "cnn_stage_num": float(cnn_stage_num_max),
            "cnn_layer_num_per_stage": float(cnn_layer_num_per_stage_max),
            "conv_channel_dim": float(conv_channel_dim_max),
            "kernel_size": float(kernel_size_max),
            # MLP part
            "hidden_num": float(hidden_num_max),
            "linear_size": float(hidden_size_max)
        }
        mode_conv: List[Dict[str, float]] = _generate_mode_for_size(param_specified_conv)
        # Implement the calculation for the maximum memory size and update
        global_structure_size, local_structure_size, encode_memory_size, \
            shared_element_buffer_size, arange_tensor_size = \
            ModelUtils.update_max_memory_size(global_structure_size,
                                           local_structure_size,
                                           encode_memory_size,
                                           shared_element_buffer_size,
                                           arange_tensor_size,
                                           mode_conv)
    return global_structure_size, local_structure_size, \
        encode_memory_size, shared_element_buffer_size, arange_tensor_size

@jit_script
def _retrieve_encode_input_minmax(param_specified: Dict[str, float],
                                 param_sampled: Dict[str, List[float]]
                                 ) -> Tuple[torch.Tensor,
                                            torch.Tensor,
                                            torch.Tensor]:
    '''
    Retrieve the min and max values for the input for encoding, including
        global and local structures, and the index encoding.
    param_specified: specified parameters for the model.
    param_sampled: distribution of the parameters for sampling.
    Returns:
        - global_structure_minmax: min and max values for the global structure.
        - local_structure_minmax: min and max values for the local structure.
        - index_encoding_minmax: min and max values for the index encoding.
    '''
    global_structure_minmax, local_structure_minmax, index_encoding_minmax = \
        ModelUtils.retrieve_encode_input_minmax(param_specified, param_sampled)
    # --- Global structure minmax ---
    # ModelUtils type
    global_structure_minmax[:, 0] = float(ModelType.CNN.value)
    # Task type
    global_structure_minmax[:, 1] = float(param_specified["task_type"])
    # Dataset type
    global_structure_minmax[:, 2] = float(param_specified["dataset_type"])
    # Number of layers
    # Hidden layers num
    hidden_num_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "min")
    hidden_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "max")        
    # Hidden size
    hidden_size_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                    "linear_size", "min")
    hidden_size_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "linear_size", "max")
    # CNN stage num
    cnn_stage_num_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                  "cnn_stage_num", "min")
    cnn_stage_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                  "cnn_stage_num", "max")        
    # CNN layer num per stage
    cnn_layer_num_per_stage_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                            "cnn_layer_num_per_stage", "min")
    cnn_layer_num_per_stage_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                            "cnn_layer_num_per_stage", "max")
    # Convolution layer num
    if cnn_stage_num_min == 0:
        num_conv_layers_min: int = 0
    else:
        num_conv_layers_min: int = (cnn_stage_num_min - 1) * cnn_layer_num_per_stage_min + 1
    if cnn_stage_num_max == 0:
        num_conv_layers_max: int = 0
    else:
        num_conv_layers_max: int = (cnn_stage_num_max - 1) * cnn_layer_num_per_stage_max + 1
    num_layer_min: int = num_conv_layers_min + hidden_num_min + 1
    num_layer_max: int = num_conv_layers_max + hidden_num_max + 1
    global_structure_minmax[0, 3] = float(num_layer_min)
    global_structure_minmax[1, 3] = float(num_layer_max)
    # CNN stage num
    global_structure_minmax[0, 4] = float(cnn_stage_num_min)
    global_structure_minmax[1, 4] = float(cnn_stage_num_max)
    # --- Local structure minmax ---
    # Layer idx
    local_structure_minmax[0, 0] = 0.0
    local_structure_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type, always has linear output
    local_structure_minmax[0, 1] = float(LayerType.LINEAR.value)
    if num_conv_layers_max > 0:
        local_structure_minmax[1, 1] = float(LayerType.CONV.value)
    else:
        local_structure_minmax[1, 1] = float(LayerType.LINEAR.value)
    # Bias type, always assume the possiblity of bias
    local_structure_minmax[0, 2] = float(BiasType.NONE.value)
    local_structure_minmax[1, 2] = float(BiasType.WITH_BIAS.value)
    # Norm type, always assume the possiblity of norm
    local_structure_minmax[0, 3] = float(NormType.NONE.value)
    local_structure_minmax[1, 3] = float(NormType.ACTIVATION_NORM.value)
    # Shortcut type, always assume the possiblity of shortcut
    local_structure_minmax[0, 4] = float(ShortcutType.NONE.value)
    local_structure_minmax[1, 4] = float(ShortcutType.STRAIGHT.value)
    # Output / input size / channel dim
    model_output_size: int = int(param_specified["output_size"])  
    min_output_size: int = model_output_size
    max_output_size: int = model_output_size
    input_size: int = int(param_specified["input_size"])
    if num_conv_layers_max > 0:
        # With convolutional layers
        # Output channel dim
        min_conv_channel_dim: int = ModelUtils.sample_int_param(param_specified, param_sampled,
            "conv_channel_dim", "min")
        max_stage_pooling_times: int = max(0, cnn_stage_num_max - 2)
        max_conv_channel_dim: int = int(ModelUtils.sample_int_param(param_specified, param_sampled, 
            "conv_channel_dim", "max") * (2 ** max_stage_pooling_times))
        min_output_size = min(min_output_size, min_conv_channel_dim)
        max_output_size = max(max_output_size, max_conv_channel_dim)
        # Input channel dim
        input_channel_dim: int = int(param_specified["input_channel_dim"])
        min_input_size: int = min(input_channel_dim, min_conv_channel_dim)
        max_input_size: int = max(input_channel_dim, max_conv_channel_dim)
        if num_conv_layers_min == 0:
            # If there could be no CNN layer, MLP input size could be the base input size
            min_input_size = min(min_input_size, input_size)
            max_input_size = max(max_input_size, input_size)
    else:
        # No convolutional layers, only MLP
        min_input_size: int = input_size
        max_input_size: int = input_size
    if hidden_num_max > 0:
        # If there could be MLP hidden layers, consider the output and input size of MLP hidden layers
        # Output size
        min_output_size = min(min_output_size, hidden_size_min)
        max_output_size = max(max_output_size, hidden_size_max)
        # Input size
        min_input_size = min(min_input_size, hidden_size_min)
        max_input_size = max(max_input_size, hidden_size_max)
    # Fill the minmax of output size
    local_structure_minmax[0, 5] = float(min_output_size)
    local_structure_minmax[1, 5] = float(max_output_size)
    # Fill the minmax of input size
    local_structure_minmax[0, 6] = float(min_input_size)
    local_structure_minmax[1, 6] = float(max_input_size)
    # Activation type, always assume all possibilities of activation
    local_structure_minmax[0, 7] = float(ActivationType.NONE.value)
    local_structure_minmax[1, 7] = float(ActivationType.SILU.value)
    # Activation param, notice that none activation param is always 0.0
    local_structure_minmax[0, 8] = 0.0
    local_structure_minmax[1, 8] = ModelUtils.sample_float_param(param_specified,
        param_sampled, "activation_param", "max")
    # Dropout rate
    local_structure_minmax[0, 9] = 0.0
    local_structure_minmax[1, 9] = ModelUtils.sample_float_param(param_specified, param_sampled,
        "dropout_rate", "max")
    # Input pooling or reshape type, always assume all possibilities of reshape
    local_structure_minmax[0, 10] = float(InputPoolingReshapeType.NONE.value)
    local_structure_minmax[1, 10] = float(InputPoolingReshapeType.FIRST_TOKEN.value)
    # Group num, linear layer always has 0 group num
    local_structure_minmax[0, 11] = 0.0
    local_structure_minmax[1, 11] = float(ModelUtils.sample_int_param(param_specified, param_sampled,
        "group_num", "max"))
    # Kernel size, linear layer always has 0 kernel size
    local_structure_minmax[0, 12] = 0.0
    max_kernel_size: int = ModelUtils.sample_int_param(param_specified, param_sampled,
        "kernel_size", "max")
    local_structure_minmax[1, 12] = float(max_kernel_size)
    # Stage wise pooling type, always assume all possibilities of pooling
    local_structure_minmax[0, 13] = float(StageWisePoolingType.NONE.value)
    local_structure_minmax[1, 13] = float(StageWisePoolingType.CONV_POOLING.value)
    # Initialization type, always assume default initialization
    local_structure_minmax[0, 20] = float(InitializationType.DEFAULT.value)
    local_structure_minmax[1, 20] = float(InitializationType.DEFAULT.value)
    # --- Index encoding minmax ---
    # Layer idx
    index_encoding_minmax[0, 0] = 0.0
    index_encoding_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type, always has linear output
    index_encoding_minmax[0, 1] = float(LayerType.LINEAR.value)
    if num_conv_layers_max > 0:
        index_encoding_minmax[1, 1] = float(LayerType.CONV.value)
    else:
        index_encoding_minmax[1, 1] = float(LayerType.LINEAR.value)
    # Param type, weights and biases
    index_encoding_minmax[0, 2] = float(ParamType.Linear.WEIGHTS.value)
    index_encoding_minmax[1, 2] = float(ParamType.Linear.BIASES.value)
    # Output idx, start from 0 to max output size - 1
    index_encoding_minmax[0, 3] = 0.0
    index_encoding_minmax[1, 3] = float(max_output_size - 1)
    # Input idx, start from -1 to max input size - 1
    index_encoding_minmax[0, 4] = -1.0
    index_encoding_minmax[1, 4] = float(max_input_size - 1)
    # Kernel H idx, start from -1 to max kernel size - 1
    index_encoding_minmax[0, 5] = -1.0
    index_encoding_minmax[1, 5] = float(max_kernel_size - 1)
    # Kernel W idx, start from -1 to max kernel size - 1
    index_encoding_minmax[0, 6] = -1.0
    index_encoding_minmax[1, 6] = float(max_kernel_size - 1)
    
    return global_structure_minmax, local_structure_minmax, index_encoding_minmax

@jit_script
def _apply_weights(x: torch.Tensor, 
                  local_mode: List[Dict[str, float]],
                  layers_params: List[List[torch.Tensor]], 
                  layers_param_shapes: List[List[List[int]]],
                  training: bool
                  ) -> torch.Tensor:
    '''
    Apply the model weights to the input tensor.
    '''
    return ModelUtils.apply_weights(x=x,
                                    local_mode=local_mode,
                                    layers_params=layers_params,
                                    layers_param_shapes=layers_param_shapes,
                                    training=training)

class CNNModelUtils:
    '''
    CNN model encoding & forwarding and modularization
    '''
    get_model_mode = staticmethod(_get_model_mode)
    generate_mode = staticmethod(_generate_mode)
    generate_mode_for_size = staticmethod(_generate_mode_for_size)
    retrieve_max_memory_size = staticmethod(_retrieve_max_memory_size)
    retrieve_encode_input_minmax = staticmethod(_retrieve_encode_input_minmax)
    apply_weights = staticmethod(_apply_weights)
