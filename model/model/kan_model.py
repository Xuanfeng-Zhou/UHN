'''
KAN Model encoding & forwarding and modularization
'''
import torch
from .model import ModelUtils, ModelType
from ..layer import KANLayerUtils
from ..layer.layer import LayerType, ActivationType, BiasType, ParamType, InitializationType
from typing import List, Dict, Tuple
from optimization import jit_script

@jit_script
def _get_model_mode(task_type: int,
                     dataset_type: int,
                     num_layers: int
                     ) -> Dict[str, float]:
    '''
    Get a model mode dictionary for KAN model.
    '''
    model_mode: Dict[str, float] = ModelUtils.get_model_mode()
    model_mode.update({
        "model_type": float(ModelType.KAN.value),
        "task_type": float(task_type),
        "dataset_type": float(dataset_type),
        "num_layers": float(num_layers)
    })
    return model_mode

@jit_script
def _generate_mode(param_specified: Dict[str, float],
                  param_sampled: Dict[str, List[float]]
                  ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    '''
    Generate a model mode dictionary for KAN model.
    Params should be specified only:
        - task_type (categorical)
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
    global_mode, local_mode = ModelUtils.generate_mode(param_specified, param_sampled)
    # --------------------- Generate global mode ---------------------
    # Hidden layers num
    hidden_num: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "minmax")
    num_layers: int = hidden_num + 1
    # Generate model mode
    global_mode.update(_get_model_mode(
        int(param_specified["task_type"]),
        int(param_specified["dataset_type"]),
        num_layers
    ))
    # --------------------- Generate local mode ---------------------
    input_size: int = int(param_specified["input_size"])
    for layer_idx in range(num_layers):
        # Sample grid size and spline order
        grid_size: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                  "grid_size", "minmax")
        spline_order: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "spline_order", "minmax")
        if layer_idx == num_layers - 1:
            output_size: int = int(param_specified["output_size"])
            # Always bias for the last layer
            bias_type: int = BiasType.WITH_BIAS.value
        else:
            output_size = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "linear_size", "minmax")
            # Bias type
            bias_type = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                 "bias_type", "multinomial")
        # Activation function
        if layer_idx == 0:
            # First layer uses no activation 
            activation_type: int = ActivationType.NONE.value
            activation_param: float = 0.0
        else:
            activation_type = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                       "activation_type", "multinomial")
            activation_param = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                          "activation_param", "minmax")
        layer_mode: Dict[str, float] = KANLayerUtils.get_layer_mode(
            layer_idx=layer_idx,
            bias_type=bias_type,
            output_size=output_size,
            input_size=input_size,
            base_activation_type=activation_type,
            base_activation_param=activation_param,
            grid_size=grid_size,
            spline_order=spline_order,
            initialization_type=InitializationType.DEFAULT.value
        )
        # Update input size
        input_size = output_size
        # Append to local mode
        local_mode.append(layer_mode)
    return global_mode, local_mode

@jit_script
def _generate_mode_for_size(param_specified: Dict[str, float]
                           ) -> List[Dict[str, float]]:
    '''
    Generate a slim version of structures for KAN model, only for model size calculation.
    Params should be specified:
        - hidden_num
        - input_size
        - grid_size
        - spline_order
        - output_size
        - linear_size
    '''
    local_mode = ModelUtils.generate_mode_for_size(param_specified)
    # Hidden layers num
    hidden_num: int = int(param_specified["hidden_num"])
    num_layers: int = hidden_num + 1
    # --------------------- Generate local mode ---------------------
    input_size: int = int(param_specified["input_size"])
    for layer_idx in range(num_layers):
        # Ignore most attributes but leave the out & in size remained
        # Always use bias in this case
        bias_type: int = BiasType.WITH_BIAS.value
        # Sample grid size and spline order
        grid_size: int = int(param_specified["grid_size"])
        spline_order: int = int(param_specified["spline_order"])
        if layer_idx == num_layers - 1:
            output_size: int = int(param_specified["output_size"])
        else:
            output_size = int(param_specified["linear_size"])
        # Generate the layer mode for kan layer, and only fill size related fields
        layer_mode: Dict[str, float] = KANLayerUtils.get_layer_mode(
            layer_idx=layer_idx,
            bias_type=bias_type,
            output_size=output_size,
            input_size=input_size,
            base_activation_type=ActivationType.NONE.value,
            base_activation_param=0.0,
            grid_size=grid_size,
            spline_order=spline_order,
            initialization_type=InitializationType.DEFAULT.value
        )
        # Update input size
        input_size = output_size
        # Append to local mode
        local_mode.append(layer_mode)
    return local_mode

@jit_script
def _retrieve_max_memory_size(param_specified: Dict[str, float],
                             param_sampled: Dict[str, List[float]]
                             ) -> Tuple[int, int, int, int, int]:
    '''
    Retrieve the maximum should-be pre-allocated memory size for encoding,
        shared elements and arange tensor for KAN model.
    '''
    global_structure_size, local_structure_size, encode_memory_size, \
        shared_element_buffer_size, arange_tensor_size = \
        ModelUtils.retrieve_max_memory_size(param_specified, param_sampled)
    
    # Hidden layers num
    hidden_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                   "hidden_num", "max")
    # Grid size and spline order
    grid_size_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                 "grid_size", "max")
    spline_order_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                   "spline_order", "max")
    # Hidden size
    hidden_size_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                   "linear_size", "max")
    
    param_specified_max: Dict[str, float] = {
        "hidden_num": float(hidden_num_max),
        "input_size": param_specified["input_size"],
        "grid_size": float(grid_size_max),
        "spline_order": float(spline_order_max),
        "output_size": param_specified["output_size"],
        "linear_size": float(hidden_size_max)
    }
    mode_max: List[Dict[str, float]] = _generate_mode_for_size(param_specified_max)
    # Implement the calculation for the maximum memory size and update
    global_structure_size, local_structure_size, encode_memory_size, \
        shared_element_buffer_size, arange_tensor_size = \
        ModelUtils.update_max_memory_size(global_structure_size,
                                       local_structure_size,
                                       encode_memory_size,
                                       shared_element_buffer_size,
                                       arange_tensor_size,
                                       mode_max)
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
    global_structure_minmax[:, 0] = float(ModelType.KAN.value)
    # Task type
    global_structure_minmax[:, 1] = float(param_specified["task_type"])
    # Dataset type
    global_structure_minmax[:, 2] = float(param_specified["dataset_type"])
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
    # Layer num
    num_layer_min: int = hidden_num_min + 1
    num_layer_max: int = hidden_num_max + 1
    global_structure_minmax[0, 3] = float(num_layer_min)
    global_structure_minmax[1, 3] = float(num_layer_max)
    # --- Local structure minmax ---
    # Layer idx
    local_structure_minmax[0, 0] = 0.0
    local_structure_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type
    local_structure_minmax[:, 1] = float(LayerType.KAN.value)
    # Bias type, always assume the possiblity of bias
    local_structure_minmax[0, 2] = float(BiasType.NONE.value)   
    local_structure_minmax[1, 2] = float(BiasType.WITH_BIAS.value)
    # Output / input channel dim
    model_output_size: int = int(param_specified["output_size"])
    model_input_size: int = int(param_specified["input_size"])        
    min_output_size: int = model_output_size
    max_output_size: int = model_output_size
    min_input_size: int = model_input_size
    max_input_size: int = model_input_size
    if hidden_num_max > 0:
        # If there could be hidden layers, consider the size of hidden layers
        # Output size
        min_output_size = min(min_output_size, hidden_size_min)
        max_output_size = max(max_output_size, hidden_size_max)
        # Input size
        min_input_size = min(min_input_size, hidden_size_min)
        max_input_size = max(max_input_size, hidden_size_max)
    local_structure_minmax[0, 5] = float(min_output_size)
    local_structure_minmax[1, 5] = float(max_output_size)
    local_structure_minmax[0, 6] = float(min_input_size)
    local_structure_minmax[1, 6] = float(max_input_size)
    # Activation type, always assume all possibilities of activation
    local_structure_minmax[0, 7] = float(ActivationType.NONE.value)
    local_structure_minmax[1, 7] = float(ActivationType.SILU.value)
    # Activation param, notice that none activation param is always 0.0
    local_structure_minmax[0, 8] = min(0.0, ModelUtils.sample_float_param(param_specified, 
        param_sampled, "activation_param", "min"))
    local_structure_minmax[1, 8] = max(0.0, ModelUtils.sample_float_param(param_specified,
        param_sampled, "activation_param", "max"))
    # Grid size
    local_structure_minmax[0, 18] = float(ModelUtils.sample_int_param(param_specified, param_sampled,
        "grid_size", "min"))
    max_grid_size: int = ModelUtils.sample_int_param(param_specified, param_sampled,
        "grid_size", "max")
    local_structure_minmax[1, 18] = float(max_grid_size)
    # Spline order
    local_structure_minmax[0, 19] = float(ModelUtils.sample_int_param(param_specified, param_sampled,
        "spline_order", "min"))
    max_spline_order: int = ModelUtils.sample_int_param(param_specified, param_sampled,
        "spline_order", "max")
    local_structure_minmax[1, 19] = float(max_spline_order)
    # Initialization type, always assume default initialization
    local_structure_minmax[0, 20] = float(InitializationType.DEFAULT.value)
    local_structure_minmax[1, 20] = float(InitializationType.DEFAULT.value)
    # --- Index encoding minmax ---
    # Layer idx
    index_encoding_minmax[0, 0] = 0.0
    index_encoding_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type
    index_encoding_minmax[:, 1] = float(LayerType.KAN.value)
    # Param type, always assume all possibilities of param type
    index_encoding_minmax[0, 2] = float(ParamType.KAN.BASE_WEIGHTS.value)
    index_encoding_minmax[1, 2] = float(ParamType.KAN.GRID_KNOTS.value)
    # Output idx, start from -1 to max output size - 1
    index_encoding_minmax[0, 3] = -1.0
    index_encoding_minmax[1, 3] = float(max_output_size - 1)
    # Input idx, start from -1 to max input size - 1
    index_encoding_minmax[0, 4] = -1.0
    index_encoding_minmax[1, 4] = float(max_input_size - 1)
    # Grid idx, start from -1 to max grid size + 2 * max spline order + 1 - 1
    index_encoding_minmax[0, 9] = -1.0
    index_encoding_minmax[1, 9] = float(max_grid_size + 2 * max_spline_order + 1 - 1)
    
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

class KANModelUtils:
    '''
    KAN model encoding & forwarding and modularization
    '''
    get_model_mode = staticmethod(_get_model_mode)
    generate_mode = staticmethod(_generate_mode)
    generate_mode_for_size = staticmethod(_generate_mode_for_size)
    retrieve_max_memory_size = staticmethod(_retrieve_max_memory_size)
    retrieve_encode_input_minmax = staticmethod(_retrieve_encode_input_minmax)
    apply_weights = staticmethod(_apply_weights)
