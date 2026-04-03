'''
Recursive Model encoding & forwarding and modularization
'''
import torch
from .model import Model, ModelUtils, ModelType
from ..layer import LinearLayerUtils, MultiheadAttentionLayer, MultiheadAttentionLayerUtils, \
    layer_apply_params_1i
from ..layer.layer import LayerType, ShortcutType, ActivationType, NormType, \
    InputPoolingReshapeType, BiasType, ParamType, InitializationType
from typing import List, Dict, Tuple
from optimization import jit_script

@jit_script
def _get_model_mode(task_type: int,
                   dataset_type: int,
                   num_layers: int,
                   num_structure_freqs: int,
                   num_index_freqs: int
                   ) -> Dict[str, float]:
    '''
    Get a model mode dictionary for Recursive model.
    '''
    model_mode: Dict[str, float] = ModelUtils.get_model_mode()
    model_mode.update({
        "model_type": float(ModelType.RECURSIVE.value),
        "task_type": float(task_type),
        "dataset_type": float(dataset_type),
        "num_layers": float(num_layers),
        "num_structure_freqs": float(num_structure_freqs),
        "num_index_freqs": float(num_index_freqs)
    })
    return model_mode

@jit_script
def _generate_mode(param_specified: Dict[str, float],
                  param_sampled: Dict[str, List[float]]
                  ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    '''
    Generate a model mode dictionary for Recursive model.
    Params should be specified only:
        - task_type (categorical)
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
    global_mode, local_mode = ModelUtils.generate_mode(param_specified, param_sampled)
    # --------------------- Generate global mode ---------------------
    # Hidden layers num
    hidden_num: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                               "hidden_num", "minmax")
    # Layer num (structure encoder + hidden layers + output layer)
    # Structure encoder has 1 MHA + 2 FFN layers + 2 FFN layers after mean along structure dim
    num_structure_encoder_layers: int = 5
    num_layers: int = num_structure_encoder_layers + hidden_num + 1
    num_structure_freqs: int = int(param_specified["num_structure_freqs"])
    num_index_freqs: int = int(param_specified["num_index_freqs"])
    # Generate model mode
    global_mode.update(_get_model_mode(
        task_type=int(param_specified["task_type"]),
        dataset_type=int(param_specified["dataset_type"]),
        num_layers=num_layers,
        num_structure_freqs=num_structure_freqs,
        num_index_freqs=num_index_freqs
    ))
    # --------------------- Generate local mode ---------------------
    # The input size of the first layer after structure encoder
    input_size: int = 2 * num_index_freqs
    # Use one hidden size for simplicity
    hidden_size: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                        "linear_size", "minmax")
    for layer_idx in range(num_layers):
        if layer_idx < num_structure_encoder_layers:
            # Structure encoder (1 MHA + 2 FFN + 2 FFN)
            embedding_dim: int = 2 * num_structure_freqs
            bias_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                   "bias_type", "multinomial")

            dropout_rate: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                        "dropout_rate", "minmax")
            if layer_idx == 0:
                # Multihead attention layer
                # Use no norm nor activation for the MHA layer in the first encoder
                norm_type: int = NormType.NONE.value
                activation_type: int = ActivationType.NONE.value
                activation_param: float = 0.0    
                shortcut_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                        "shortcut_type", "multinomial")                
                num_heads: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "num_heads", "minmax")
                # Generate the layer mode
                layer_mode: Dict[str, float] = MultiheadAttentionLayerUtils.get_layer_mode(
                    layer_idx=layer_idx,
                    bias_type=bias_type,
                    norm_type=norm_type,
                    shortcut_type=shortcut_type,
                    embedding_dim=embedding_dim,
                    activation_type=activation_type,
                    activation_param=activation_param,
                    dropout_rate=dropout_rate,
                    num_heads=num_heads,
                    initialization_type=InitializationType.DEFAULT.value
                )
            elif layer_idx < 3:
                # First 2 feedforward layer
                norm_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                   "norm_type", "multinomial")
                activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                       "activation_type", "multinomial")
                activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                           "activation_param", "minmax")
                shortcut_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                        "shortcut_type", "multinomial")                
                # Generate the layer mode
                layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                    layer_idx=layer_idx,
                    bias_type=bias_type,
                    norm_type=norm_type,
                    shortcut_type=shortcut_type,
                    output_size=embedding_dim,
                    input_size=embedding_dim,
                    activation_type=activation_type,
                    activation_param=activation_param,
                    dropout_rate=dropout_rate,
                    input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                    initialization_type=InitializationType.DEFAULT.value
                )
            else:
                # Last 2 feedforward layer after mean along structure dim, 
                #   mean along structure -> linear -> activation -> linear
                # No norm nor shortcut
                norm_type: int = NormType.NONE.value
                shortcut_type: int = ShortcutType.NONE.value
                if layer_idx == 3:
                    # First layer of the second FFN use no activation
                    activation_type: int = ActivationType.NONE.value
                    activation_param: float = 0.0
                    # Generate the layer mode
                    layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                        layer_idx=layer_idx,
                        bias_type=bias_type,
                        norm_type=norm_type,
                        shortcut_type=shortcut_type,
                        output_size=hidden_size,
                        input_size=embedding_dim,
                        activation_type=activation_type,
                        activation_param=activation_param,
                        dropout_rate=dropout_rate,
                        input_pooling_reshape_type=InputPoolingReshapeType.AVG_SEQUENCE.value,
                        initialization_type=InitializationType.DEFAULT.value
                    )
                else:
                    # Second layer of the second FFN use activation
                    activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                            "activation_type", "multinomial")
                    activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                            "activation_param", "minmax")
                    # Generate the layer mode
                    layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                        layer_idx=layer_idx,
                        bias_type=bias_type,
                        norm_type=norm_type,
                        shortcut_type=shortcut_type,
                        output_size=hidden_size,
                        input_size=hidden_size,
                        activation_type=activation_type,
                        activation_param=activation_param,
                        dropout_rate=dropout_rate,
                        input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                        initialization_type=InitializationType.ZERO.value
                    )
        else:
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
            # Norm type and activation
            if layer_idx == num_structure_encoder_layers:
                # First layer after structure encoder uses no norm nor activation
                norm_type: int = NormType.NONE.value
                activation_type: int = ActivationType.NONE.value
                activation_param: float = 0.0
            else:
                if layer_idx == num_layers - 1:
                    # Output layer uses no norm
                    norm_type: int = NormType.NONE.value
                else:
                    norm_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                        "norm_type", "multinomial")
                activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                           "activation_type", "multinomial")
                activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                           "activation_param", "minmax")
            # Shortcut type
            if layer_idx == num_structure_encoder_layers or layer_idx == num_layers - 1:
                # First layer after structure encoder and output layer use no shortcut
                shortcut_type: int = ShortcutType.NONE.value
            else:
                shortcut_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                           "shortcut_type", "multinomial")
            # Output size
            if layer_idx == num_layers - 1:
                # Output layer output 1-dim value
                output_size: int = 1
            else:
                output_size: int = hidden_size
            # Generate the layer mode
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
                input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                initialization_type=InitializationType.DEFAULT.value
            )
            # Update the input size for the next layer
            input_size = output_size
        # Append the layer mode
        local_mode.append(layer_mode)
    return global_mode, local_mode

@jit_script
def _generate_mode_for_size(param_specified: Dict[str, float]
                           ) -> List[Dict[str, float]]:
    '''
    Generate a slim version of structures for Recursive model, only for model size calculation.
    Params should be specified:
        - hidden_num
        - num_structure_freqs
        - num_index_freqs
        - linear_size
    '''
    local_mode = ModelUtils.generate_mode_for_size(param_specified)
    # Hidden layers num
    hidden_num: int = int(param_specified["hidden_num"])
    # Layer num (structure encoder + hidden layers + output layer)
    # Structure encoder has 1 MHA + 2 FFN layers + 2 FFN layers after mean along structure dim
    num_structure_encoder_layers: int = 5
    num_layers: int = num_structure_encoder_layers + hidden_num + 1
    num_structure_freqs: int = int(param_specified["num_structure_freqs"])
    num_index_freqs: int = int(param_specified["num_index_freqs"])
    # --------------------- Generate local mode ---------------------
    # The input size of the first layer after structure encoder
    input_size: int = 2 * num_index_freqs
    # Use one hidden size for simplicity
    hidden_size: int = int(param_specified["linear_size"])
    for layer_idx in range(num_layers):
        # Ignore most attributes but leave the out & in size remained
        # Always use bias in this case
        bias_type: int = BiasType.WITH_BIAS.value                     
        if layer_idx < num_structure_encoder_layers:
            # Structure encoder (1 MHA + 2 FFN + 2 FFN)
            embedding_dim: int = 2 * num_structure_freqs
            if layer_idx == 0:
                # Multihead attention layer
                # Generate the layer mode, and only fill size related fields
                layer_mode: Dict[str, float] = MultiheadAttentionLayerUtils.get_layer_mode(
                    layer_idx=layer_idx,
                    bias_type=bias_type,
                    norm_type=NormType.NONE.value,
                    shortcut_type=ShortcutType.NONE.value,
                    embedding_dim=embedding_dim,
                    activation_type=ActivationType.NONE.value,
                    activation_param=0.0,
                    dropout_rate=0.0,
                    num_heads=0,
                    initialization_type=InitializationType.DEFAULT.value
                )

            elif layer_idx < 3:
                # First 2 feedforward layer
                # Generate the layer mode, and only fill size related fields
                layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                    layer_idx=layer_idx,
                    bias_type=bias_type,
                    norm_type=NormType.NONE.value,
                    shortcut_type=ShortcutType.NONE.value,
                    output_size=embedding_dim,
                    input_size=embedding_dim,
                    activation_type=ActivationType.NONE.value,
                    activation_param=0.0,
                    dropout_rate=0.0,
                    input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                    initialization_type=InitializationType.DEFAULT.value
                )
            else:
                # Last 2 feedforward layer after mean along structure dim, 
                #   mean along structure -> linear -> activation -> linear
                # Generate the layer mode, and only fill size related fields
                if layer_idx == 3:
                    layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                        layer_idx=layer_idx,
                        bias_type=bias_type,
                        norm_type=NormType.NONE.value,
                        shortcut_type=ShortcutType.NONE.value,
                        output_size=hidden_size,
                        input_size=embedding_dim,
                        activation_type=ActivationType.NONE.value,
                        activation_param=0.0,
                        dropout_rate=0.0,
                        input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                        initialization_type=InitializationType.DEFAULT.value
                    )
                else:
                    layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                        layer_idx=layer_idx,
                        bias_type=bias_type,
                        norm_type=NormType.NONE.value,
                        shortcut_type=ShortcutType.NONE.value,
                        output_size=hidden_size,
                        input_size=hidden_size,
                        activation_type=ActivationType.NONE.value,
                        activation_param=0.0,
                        dropout_rate=0.0,
                        input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                        initialization_type=InitializationType.DEFAULT.value
                    )
        else:   
            # Output size
            if layer_idx == num_layers - 1:
                # Output layer output 1-dim value
                output_size: int = 1
            else:
                output_size: int = hidden_size
            # Generate the layer mode, and only fill size related fields
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
        shared elements and arange tensor for Recursive model.
    '''
    global_structure_size, local_structure_size, encode_memory_size, \
        shared_element_buffer_size, arange_tensor_size = \
        ModelUtils.retrieve_max_memory_size(param_specified, param_sampled)
    
    # Hidden layers num
    hidden_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "hidden_num", "max")
    hidden_size_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "linear_size", "max")
    
    param_specified_max: Dict[str, float] = {
        "hidden_num": float(hidden_num_max),
        "num_structure_freqs": float(param_specified["num_structure_freqs"]),
        "num_index_freqs": param_specified["num_index_freqs"],
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
    global_structure_minmax[:, 0] = float(ModelType.RECURSIVE.value)
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
    num_layer_min: int = 5 + hidden_num_min + 1
    num_layer_max: int = 5 + hidden_num_max + 1
    global_structure_minmax[0, 3] = float(num_layer_min)
    global_structure_minmax[1, 3] = float(num_layer_max)
    # Structure freqs
    num_structure_freqs: int = int(param_specified["num_structure_freqs"])
    global_structure_minmax[:, 6] = float(num_structure_freqs)
    # Index freq num
    num_index_freqs: int = int(param_specified["num_index_freqs"])
    global_structure_minmax[:, 7] = float(num_index_freqs)
    # --- Local structure minmax ---
    # Layer idx
    local_structure_minmax[0, 0] = 0.0
    local_structure_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type, always assume all possibilities of layer type
    local_structure_minmax[0, 1] = float(LayerType.LINEAR.value)
    local_structure_minmax[1, 1] = float(LayerType.MHA.value)
    # Bias type, always assume the possiblity of bias
    local_structure_minmax[0, 2] = float(BiasType.NONE.value)   
    local_structure_minmax[1, 2] = float(BiasType.WITH_BIAS.value)
    # Norm type, always assume the possiblity of norm
    local_structure_minmax[0, 3] = float(NormType.NONE.value)
    local_structure_minmax[1, 3] = float(NormType.ACTIVATION_NORM.value)
    # Shortcut type, always assume all possibilities of shortcut
    local_structure_minmax[0, 4] = float(ShortcutType.NONE.value)
    local_structure_minmax[1, 4] = float(ShortcutType.STRAIGHT.value)        
    # Min or max of the output size of structure encoder attention, FFN and the whole model
    min_output_size: int = min(2 * num_structure_freqs, hidden_size_min, 1)
    max_output_size: int = max(2 * num_structure_freqs, hidden_size_max, 1)
    # The input size of the structure encoder attention, FFN and the index encoding
    min_input_size: int = min(2 * num_structure_freqs, hidden_size_min, 2 * num_index_freqs)
    max_input_size: int = max(2 * num_structure_freqs, hidden_size_max, 2 * num_index_freqs)
    local_structure_minmax[0, 5] = float(min_output_size)
    local_structure_minmax[1, 5] = float(max_output_size)
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
    # Input pooling or reshape type, none and avg sequence
    local_structure_minmax[0, 10] = float(InputPoolingReshapeType.NONE.value)
    local_structure_minmax[1, 10] = float(InputPoolingReshapeType.AVG_SEQUENCE.value)
    # Number of heads
    local_structure_minmax[0, 14] = 0.0
    local_structure_minmax[1, 14] = float(ModelUtils.sample_int_param(param_specified, param_sampled,
        "num_heads", "max"))
    # Initialization type, default and zero
    local_structure_minmax[0, 20] = float(InitializationType.DEFAULT.value)
    local_structure_minmax[1, 20] = float(InitializationType.ZERO.value)
    # --- Index encoding minmax ---
    # Layer idx
    index_encoding_minmax[0, 0] = 0.0
    index_encoding_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type
    index_encoding_minmax[0, 1] = float(LayerType.LINEAR.value)
    index_encoding_minmax[1, 1] = float(LayerType.MHA.value)
    # Param type, always assume all possibilities of param type
    index_encoding_minmax[0, 2] = float(ParamType.MHA.PROJECTION_WEIGHTS_Q.value)
    index_encoding_minmax[1, 2] = float(ParamType.MHA.OUT_PROJECTION_BIASES.value)
    # Output idx, start from 0 to max output size - 1
    index_encoding_minmax[0, 3] = 0.0
    index_encoding_minmax[1, 3] = float(max_output_size - 1)
    # Input idx, start from -1 to max input size - 1
    index_encoding_minmax[0, 4] = -1.0
    index_encoding_minmax[1, 4] = float(max_input_size - 1)
    
    return global_structure_minmax, local_structure_minmax, index_encoding_minmax

@jit_script
def _apply_weights(structure_fourier: torch.Tensor,
                  index_fourier: torch.Tensor,
                  local_mode: List[Dict[str, float]],
                  layers_params: List[List[torch.Tensor]],
                  layers_param_shapes: List[List[List[int]]],
                  training: bool
                  ) -> torch.Tensor:
    '''
    Apply the model weights to the input tensor.
    '''
    num_structure_encoder_layers: int = 5
    # Encode with local strcutre encoder
    for layer_idx in range(num_structure_encoder_layers):
        layer_type: int = int(local_mode[layer_idx]['layer_type'])
        if layer_type == LayerType.MHA.value:
            # MHA layer
            structure_fourier = MultiheadAttentionLayerUtils.apply_params(
                x=structure_fourier,
                # No padding mask
                padding_mask=None,
                layer_structure=local_mode[layer_idx],
                layer_params=layers_params[layer_idx],
                layer_param_shapes=layers_param_shapes[layer_idx],
                training=training
            )
        else:
            # Linear layer
            structure_fourier = layer_apply_params_1i(
                layer_type=layer_type,
                x=structure_fourier,
                layer_structure=local_mode[layer_idx],
                layer_params=layers_params[layer_idx],
                layer_param_shapes=layers_param_shapes[layer_idx],
                training=training
            )
    # Pass through the index fourier to rest of the layers till the second to last layer
    for layer_idx in range(num_structure_encoder_layers, len(local_mode) - 1):
        layer_type: int = int(local_mode[layer_idx]['layer_type'])
        index_fourier = layer_apply_params_1i(
            layer_type=layer_type,
            x=index_fourier,
            layer_structure=local_mode[layer_idx],
            layer_params=layers_params[layer_idx],
            layer_param_shapes=layers_param_shapes[layer_idx],
            training=training
        )
    # Combine the structure and index info
    x: torch.Tensor = structure_fourier + index_fourier
    # Pass through the last layer
    weight_value: torch.Tensor = layer_apply_params_1i(
        layer_type=int(local_mode[-1]['layer_type']),
        x=x,
        layer_structure=local_mode[-1],
        layer_params=layers_params[-1],
        layer_param_shapes=layers_param_shapes[-1],
        training=training
    )
    return weight_value.squeeze()

class RecursiveModelUtils:
    '''
    Recursive model encoding & forwarding and modularization
    '''
    get_model_mode = staticmethod(_get_model_mode)
    generate_mode = staticmethod(_generate_mode)
    generate_mode_for_size = staticmethod(_generate_mode_for_size)
    retrieve_max_memory_size = staticmethod(_retrieve_max_memory_size)
    retrieve_encode_input_minmax = staticmethod(_retrieve_encode_input_minmax)
    apply_weights = staticmethod(_apply_weights)

class RecursiveModel(Model):
    '''
    Recursive model encoding & forwarding and modularization
    '''
    def forward(self, 
                structure_fourier: torch.Tensor,
                index_fourier: torch.Tensor
                ) -> torch.Tensor:
        '''
        Forward pass of the Recursive model.
        '''
        num_structure_encoder_layers: int = 5
        # Encode with local strcutre encoder
        for layer_idx in range(num_structure_encoder_layers):
            layer: type[Model] = self.layers[layer_idx]
            if isinstance(layer, MultiheadAttentionLayer):
                # Apply MHA layer without padding mask
                structure_fourier = layer(structure_fourier, None)
            else:
                # Apply linear layer
                structure_fourier = layer(structure_fourier)
        # Pass through the index fourier to rest of the layers till the second to last layer
        for layer_idx in range(num_structure_encoder_layers, len(self.layers) - 1):
            index_fourier = self.layers[layer_idx](index_fourier)
        # Combine the structure and index info
        x: torch.Tensor = structure_fourier + index_fourier
        # Pass through the last layer
        weight_value: torch.Tensor = self.layers[-1](x)
        return weight_value.squeeze()
