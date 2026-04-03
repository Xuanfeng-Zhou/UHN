'''
GAT Model encoding & forwarding and modularization
'''
import torch
from .model import Model, ModelUtils, ModelType
from ..layer import GATLayerUtils, layer_apply_params_2i
from ..layer.layer import LayerType, ActivationType, \
    BiasType, HeadConcatType, ParamType, NormType, InitializationType
from typing import List, Dict, Tuple
from optimization import jit_script

@jit_script
def _get_model_mode(task_type: int,
                     dataset_type: int,
                     num_layers: int
                     ) -> Dict[str, float]:
    '''
    Get a model mode dictionary for GAT model.
    '''
    model_mode: Dict[str, float] = ModelUtils.get_model_mode()
    model_mode.update({
        "model_type": float(ModelType.GAT.value),
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
    Generate model and layers mode dictionaries for GAT model.
    Params should be specified only:
        - task_type (categorical)
        - dataset_type (categorical)
        - input_size (regressive)
        - output_size (regressive)
    
    Params should be either specified or sampled:
        - hidden_num (minmax)
        - num_heads_hidden (minmax)
        - linear_size_per_head (minmax)
        - num_heads_output (minmax)
        - bias_type (multinomial)
        - norm_type (multinomial)
        - activation_type (multinomial)
        - activation_param (minmax)
        - dropout_rate (minmax)
    '''
    global_mode, local_mode = ModelUtils.generate_mode(param_specified, param_sampled)
    # --------------------- Generate global mode ---------------------
    # Hidden layers num
    hidden_num: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "minmax")
    num_layers: int = hidden_num + 1
    # Generate the model mode
    global_mode.update(_get_model_mode(
        task_type=int(param_specified["task_type"]),
        dataset_type=int(param_specified["dataset_type"]),
        num_layers=num_layers
    ))
    # --------------------- Generate local mode ---------------------
    input_dim: int = int(param_specified["input_size"])
    for layer_idx in range(num_layers):
        # Bias type
        if layer_idx == num_layers - 1:
            # Output layer always has bias
            bias_type: int = BiasType.WITH_BIAS.value
        else:
            bias_type: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                      "bias_type", "multinomial")
        # Activation type
        if layer_idx == 0:
            # Input layer always has no activation nor normalization
            norm_type: int = NormType.NONE.value
            activation_type: int = ActivationType.NONE.value
            activation_param: float = 0.0
        else:
            norm_type: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                    "norm_type", "multinomial")            
            activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                           "activation_type", "multinomial")
            activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled, 
                                                                 "activation_param", "minmax")
        # Dropout rate
        dropout_rate: float = ModelUtils.sample_float_param(param_specified, param_sampled, 
                                                         "dropout_rate", "minmax")
        # Output size
        if layer_idx == num_layers - 1:
            output_dim: int = int(param_specified["output_size"])
            num_heads: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                      "num_heads_output", "minmax")
            # The output layer always uses the average head concatenation
            head_concat_type: int = HeadConcatType.AVG.value
        else:
            output_dim: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                     "linear_size_per_head", "minmax")
            num_heads: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                      "num_heads_hidden", "minmax")
            # The hidden layers always use the concatenation head concatenation
            head_concat_type: int = HeadConcatType.CONCAT.value
        # Generate the layer mode
        layer_mode: Dict[str, float] = GATLayerUtils.get_layer_mode(
            layer_idx=layer_idx,
            bias_type=bias_type,
            norm_type=norm_type,
            output_dim=output_dim,
            input_dim=input_dim,
            activation_type=activation_type,
            activation_param=activation_param,
            dropout_rate=dropout_rate,
            num_heads=num_heads,
            head_concat_type=head_concat_type,
            initialization_type=InitializationType.DEFAULT.value
        )
        # Update the input dimension
        if head_concat_type == HeadConcatType.AVG.value:
            input_dim = output_dim
        else:
            input_dim = output_dim * num_heads
        # Append the layer mode
        local_mode.append(layer_mode)
    return global_mode, local_mode

@jit_script
def _generate_mode_for_size(param_specified: Dict[str, float]
                           ) -> List[Dict[str, float]]:
    '''
    Generate a slim version of structures for GAT model, only for model size calculation.
    
    Params should be specified:
        - input_size
        - output_size
        - hidden_num
        - num_heads_hidden
        - linear_size_per_head
        - num_heads_output
    '''
    local_mode = ModelUtils.generate_mode_for_size(param_specified)
    # Hidden layers num
    hidden_num: int = int(param_specified["hidden_num"])
    num_layers: int = hidden_num + 1
    # --------------------- Generate local mode ---------------------
    input_dim: int = int(param_specified["input_size"])
    for layer_idx in range(num_layers):
        # Ignore most attributes but leave the out & in size remained
        # Always use bias in this case
        bias_type: int = BiasType.WITH_BIAS.value            
        # Output size
        if layer_idx == num_layers - 1:
            output_dim: int = int(param_specified["output_size"])
            num_heads: int = int(param_specified["num_heads_output"])
            # The output layer always uses the average head concatenation
            head_concat_type: int = HeadConcatType.AVG.value                
        else:
            output_dim: int = int(param_specified["linear_size_per_head"])
            num_heads: int = int(param_specified["num_heads_hidden"])
            # The hidden layers always use the concatenation head concatenation
            head_concat_type: int = HeadConcatType.CONCAT.value
        # Generate the layer mode for gcn layer, and only fill size related fields
        layer_mode: Dict[str, float] = GATLayerUtils.get_layer_mode(
            layer_idx=layer_idx,
            bias_type=bias_type,
            norm_type=NormType.NONE.value,
            output_dim=output_dim,
            input_dim=input_dim,
            activation_type=ActivationType.NONE.value,
            activation_param=0.0,
            dropout_rate=0.0,
            num_heads=num_heads,
            head_concat_type=head_concat_type,
            initialization_type=InitializationType.DEFAULT.value
        )
        # Update the input dimension
        if head_concat_type == HeadConcatType.AVG.value:
            input_dim = output_dim
        else:
            input_dim = output_dim * num_heads
        # Append the layer mode
        local_mode.append(layer_mode)
    return local_mode            

@jit_script
def _retrieve_max_memory_size(param_specified: Dict[str, float],
                             param_sampled: Dict[str, List[float]]
                             ) -> Tuple[int, int, int, int, int]:
    '''
    Retrieve the maximum should-be pre-allocated memory size for encoding,
        shared elements and arange tensor for GAT model.
    '''        
    global_structure_size, local_structure_size, encode_memory_size, \
        shared_element_buffer_size, arange_tensor_size = \
        ModelUtils.retrieve_max_memory_size(param_specified, param_sampled)
    
    # Hidden layers num
    hidden_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "max")      
    # Hidden layer head num
    num_heads_hidden_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                         "num_heads_hidden", "max")
    # Linear size per head
    linear_size_per_head_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                             "linear_size_per_head", "max")
    # Output layer head num
    num_heads_output_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                         "num_heads_output", "max")
    param_specified_max: Dict[str, float] = {
        "input_size": param_specified["input_size"],
        "output_size": param_specified["output_size"],
        "hidden_num": float(hidden_num_max),
        "num_heads_hidden": float(num_heads_hidden_max),
        "linear_size_per_head": float(linear_size_per_head_max),
        "num_heads_output": float(num_heads_output_max)
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
    global_structure_minmax[:, 0] = float(ModelType.GAT.value)
    # Task type
    global_structure_minmax[:, 1] = float(param_specified["task_type"])
    # Dataset type
    global_structure_minmax[:, 2] = float(param_specified["dataset_type"])
    # Hidden layers num
    hidden_num_min: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                               "hidden_num", "min")
    hidden_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "max")
    # Linear size per head
    linear_size_per_head_min: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                             "linear_size_per_head", "min")
    linear_size_per_head_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                             "linear_size_per_head", "max")
    # Hidden layer head num
    num_heads_hidden_min: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                         "num_heads_hidden", "min")
    num_heads_hidden_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                          "num_heads_hidden", "max")
    # Output layer head num
    num_heads_output_min: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                         "num_heads_output", "min")
    num_heads_output_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                         "num_heads_output", "max")
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
    local_structure_minmax[:, 1] = float(LayerType.GAT.value)
    # Bias type, always assume the possiblity of bias
    local_structure_minmax[0, 2] = float(BiasType.NONE.value)   
    local_structure_minmax[1, 2] = float(BiasType.WITH_BIAS.value)
    # Norm type, always assume the possiblity of norm
    local_structure_minmax[0, 3] = float(NormType.NONE.value)
    local_structure_minmax[1, 3] = float(NormType.ACTIVATION_NORM.value)    
    # Output / input channel dim
    model_output_size: int = int(param_specified["output_size"])
    model_input_size: int = int(param_specified["input_size"])
    min_output_size_per_head: int = model_output_size
    max_output_size_per_head: int = model_output_size
    min_input_size: int = model_input_size
    max_input_size: int = model_input_size
    if hidden_num_max > 0:
        # If there could be hidden layers, consider the size of hidden layers
        # Output size
        min_output_size_per_head = min(min_output_size_per_head, linear_size_per_head_min)
        max_output_size_per_head = max(max_output_size_per_head, linear_size_per_head_max)
        # Input size
        min_input_size = min(min_input_size, linear_size_per_head_min * num_heads_hidden_min)
        max_input_size = max(max_input_size, linear_size_per_head_max * num_heads_hidden_max)
    local_structure_minmax[0, 5] = float(min_output_size_per_head)
    local_structure_minmax[1, 5] = float(max_output_size_per_head)
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
    # Number of heads
    local_structure_minmax[0, 14] = float(min(num_heads_hidden_min, num_heads_output_min))
    local_structure_minmax[1, 14] = float(max(num_heads_hidden_max, num_heads_output_max))
    # Head concat type, always assume all possibilities of head concat
    local_structure_minmax[0, 15] = float(HeadConcatType.CONCAT.value)
    local_structure_minmax[1, 15] = float(HeadConcatType.AVG.value)
    # Initialization type, always assume default initialization
    local_structure_minmax[0, 20] = float(InitializationType.DEFAULT.value)
    local_structure_minmax[1, 20] = float(InitializationType.DEFAULT.value)
    # --- Index encoding minmax ---
    # Layer idx
    index_encoding_minmax[0, 0] = 0.0
    index_encoding_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type
    index_encoding_minmax[:, 1] = float(LayerType.GAT.value)
    # Param type, weights, biases, attention src, attention dst
    index_encoding_minmax[0, 2] = float(ParamType.GAT.WEIGHTS.value)
    index_encoding_minmax[1, 2] = float(ParamType.GAT.ATTENTION_WEIGHTS_DST.value)
    # Output idx, start from 0 to max output size * max head num - 1
    index_encoding_minmax[0, 3] = 0.0
    index_encoding_minmax[1, 3] = float(max(linear_size_per_head_max * num_heads_hidden_max,
        model_output_size * num_heads_output_max) - 1)
    # Input idx, start from -1 to max input size - 1
    index_encoding_minmax[0, 4] = -1.0
    index_encoding_minmax[1, 4] = float(max_input_size - 1)
    
    return global_structure_minmax, local_structure_minmax, index_encoding_minmax

@jit_script
def _apply_weights(x: torch.Tensor, 
                  edge_index: torch.Tensor,
                  local_mode: List[Dict[str, float]],
                  layers_params: List[List[torch.Tensor]], 
                  layers_param_shapes: List[List[List[int]]],
                  training: bool
                  ) -> torch.Tensor:
    '''
    Apply the model weights to the input tensor.
    '''
    # Apply the weights for each layer
    for layer_idx in range(len(local_mode)):
        layer_type: int = int(local_mode[layer_idx]['layer_type'])
        x = layer_apply_params_2i(
            layer_type=layer_type,
            x=x,
            y=edge_index,
            layer_structure=local_mode[layer_idx],
            layer_params=layers_params[layer_idx],
            layer_param_shapes=layers_param_shapes[layer_idx],
            training=training
        )             
    return x

class GATModelUtils:
    '''
    GAT model encoding & forwarding and modularization
    '''
    get_model_mode = staticmethod(_get_model_mode)
    generate_mode = staticmethod(_generate_mode)
    generate_mode_for_size = staticmethod(_generate_mode_for_size)
    retrieve_max_memory_size = staticmethod(_retrieve_max_memory_size)
    retrieve_encode_input_minmax = staticmethod(_retrieve_encode_input_minmax)
    apply_weights = staticmethod(_apply_weights)

class GATModel(Model):
    '''
    GAT model encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        '''
        Forward the input tensor through the GAT model.
        '''
        # Forward the input tensor through the GAT layers
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
