'''
Transformer model encoding & forwarding and modularization
'''
import torch
from .model import Model, ModelUtils, ModelType
from ..layer import MultiheadAttentionLayer, LinearLayerUtils, MultiheadAttentionLayerUtils, \
    EmbeddingLayerUtils, layer_apply_params_1i
from ..layer.layer import LayerType, ShortcutType, ActivationType, NormType, \
    InputPoolingReshapeType, BiasType, ParamType, InitializationType
from typing import List, Dict, Tuple, Optional
from optimization import jit_script

@jit_script
def _get_model_mode(task_type: int,
                     dataset_type: int,
                     num_layers: int,
                     num_encoders: int,
                     ) -> Dict[str, float]:
    '''
    Get a model mode dictionary for Transformer model.
    '''
    model_mode: Dict[str, float] = ModelUtils.get_model_mode()
    model_mode.update({
        "model_type": float(ModelType.TRANSFORMER.value),
        "task_type": float(task_type),
        "dataset_type": float(dataset_type),
        "num_layers": float(num_layers),
        "num_encoders": float(num_encoders),
    })
    return model_mode

@jit_script
def _generate_mode(param_specified: Dict[str, float],
                  param_sampled: Dict[str, List[float]]
                  ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    '''
    Generate a model mode dictionary for Transformer model.
    Params should be specified only:
        - task_type (categorical)
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
    global_mode, local_mode = ModelUtils.generate_mode(param_specified, param_sampled)
    # --------------------- Generate global mode ---------------------
    # Encoder num
    num_encoders: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                            "encoder_num", "minmax")    
    # Encoder transition layer idx
    encoder_transition_layer_idx: List[int] = []
    # The first transition idx is right after the embedding layer
    encoder_transition_layer_idx.append(1)
    for encoder_idx in range(num_encoders):
        # Sample the number of linear layers in each encoder
        encoder_linear_layer_num: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "linear_layer_num_per_encoder", "minmax",
                                                check_idx=True, attribute_idx=encoder_idx)
        encoder_layer_num: int = encoder_linear_layer_num + 1
        # Append the transition layer idx
        encoder_transition_layer_idx.append(encoder_transition_layer_idx[-1] + \
                                            encoder_layer_num)

    # Layer num, embedding layer + encoder layers + output layer
    num_layers: int = encoder_transition_layer_idx[-1] + 1
    # Generate the model mode
    global_mode.update(_get_model_mode(
        task_type=int(param_specified["task_type"]),
        dataset_type=int(param_specified["dataset_type"]),
        num_layers=num_layers,
        num_encoders=num_encoders
    ))
    # --------------------- Generate local mode ---------------------
    # Get vocabulary size
    vocab_size: int = int(param_specified["vocab_size"])
    # Sample the first head num
    num_heads: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                            "num_heads", "minmax")
    # Sample the first embedding dimension
    embedding_dim: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "embedding_dim", "minmax",
                                                divisable_by=num_heads)    
    # Max sequence length
    max_sequence_length: int = int(param_specified["max_sequence_length"])    
    for layer_idx in range(num_layers):
        if layer_idx == 0:
            # Embedding layer
            dropout_rate: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                        "dropout_rate", "minmax")
            layer_mode: Dict[str, float] = EmbeddingLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                embedding_num=vocab_size,
                embedding_dim=embedding_dim,
                max_sequence_length=max_sequence_length,
                dropout_rate=dropout_rate,
                initialization_type=InitializationType.DEFAULT.value
            )
        elif layer_idx < num_layers - 1:
            # Encoder layers
            bias_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                   "bias_type", "multinomial")
            shortcut_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                       "shortcut_type", "multinomial")
            dropout_rate: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                        "dropout_rate", "minmax")
            if layer_idx in encoder_transition_layer_idx:
                # Multihead attention layer
                if layer_idx == 1:
                    # Use no norm nor activation for the MHA layer in the first encoder
                    norm_type: int = NormType.NONE.value
                    activation_type: int = ActivationType.NONE.value
                    activation_param: float = 0.0    
                else:
                    # Use norm and activation for the MHA layer in the rest encoders
                    norm_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                        "norm_type", "multinomial")
                    activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                            "activation_type", "multinomial")
                    activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                                "activation_param", "minmax")
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
            else:
                # Linear layer
                norm_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "norm_type", "multinomial")
                activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                        "activation_type", "multinomial")
                activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                                "activation_param", "minmax")
                # Output size of the linear layer
                if shortcut_type == ShortcutType.NONE.value:
                    # If next layer is a MHA layer, make sure the output size is divisible by num_heads
                    if (layer_idx + 1 < num_layers - 1) and \
                        ((layer_idx + 1) in encoder_transition_layer_idx):
                        num_heads = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                            "num_heads", "minmax")                    
                        output_size: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                            "embedding_dim", "minmax",
                                                            divisable_by=num_heads)
                    else:
                        output_size: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                            "embedding_dim", "minmax")
                else:
                    output_size: int = embedding_dim
                # Generate the layer mode
                layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                    layer_idx=layer_idx,
                    bias_type=bias_type,
                    norm_type=norm_type,
                    shortcut_type=shortcut_type,
                    output_size=output_size,
                    input_size=embedding_dim,
                    activation_type=activation_type,
                    activation_param=activation_param,
                    dropout_rate=dropout_rate,
                    input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                    initialization_type=InitializationType.DEFAULT.value
                )
                # Update the embedding dimension
                embedding_dim = output_size
        else:
            # Output layer
            norm_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "norm_type", "multinomial")
            activation_type: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                    "activation_type", "multinomial")
            activation_param: float = ModelUtils.sample_float_param(param_specified, param_sampled,
                                                            "activation_param", "minmax")
            output_size: int = int(param_specified["output_size"])
            # Generate the layer mode
            layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                bias_type=BiasType.WITH_BIAS.value,
                norm_type=NormType.NONE.value,
                shortcut_type=ShortcutType.NONE.value,
                output_size=output_size,
                input_size=embedding_dim,
                activation_type=ActivationType.NONE.value,
                activation_param=0.0,
                dropout_rate=0.0,
                input_pooling_reshape_type=InputPoolingReshapeType.FIRST_TOKEN.value,
                initialization_type=InitializationType.DEFAULT.value
            )
            # Update the embedding dimension
            embedding_dim = output_size                
        local_mode.append(layer_mode)
    return global_mode, local_mode

@jit_script
def _generate_mode_for_size(param_specified: Dict[str, float]
                           ) -> List[Dict[str, float]]:
    '''
    Generate a slim version of structures for Transformer model, only for model size calculation.
    
    Params should be specified:
        - encoder_num
        - linear_layer_num_per_encoder
        - vocab_size
        - embedding_dim
        - output_size
        - max_sequence_length
    '''
    local_mode = ModelUtils.generate_mode_for_size(param_specified)
    # Encoder num
    num_encoders: int = int(param_specified["encoder_num"])
    # Linear layer num in each encoder
    num_linear_layer_per_encoder: int = int(param_specified["linear_layer_num_per_encoder"])
    # Layer num, embedding layer + encoder layers + output layer
    num_layers: int = 1 + num_encoders * (num_linear_layer_per_encoder + 1) + 1
    # --------------------- Generate local mode ---------------------
    # Get vocabulary size
    vocab_size: int = int(param_specified["vocab_size"])
    # Embedding dimension
    embedding_dim: int = int(param_specified["embedding_dim"])
    # Max sequence length
    max_sequence_length: int = int(param_specified["max_sequence_length"])
    for layer_idx in range(num_layers):
        # Ignore most attributes but leave the out & in size remained
        # Always use bias in this case
        bias_type: int = BiasType.WITH_BIAS.value         
        if layer_idx == 0:
            # Embedding layer
            layer_mode: Dict[str, float] = EmbeddingLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                embedding_num=vocab_size,
                embedding_dim=embedding_dim,
                max_sequence_length=max_sequence_length,
                dropout_rate=0.0,
                initialization_type=InitializationType.DEFAULT.value
            )
        elif layer_idx < num_layers - 1:
            # Encoder layers
            if (layer_idx - 1) % (num_linear_layer_per_encoder + 1) == 0:
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
            else:
                # Linear layer
                # Output size of the linear layer
                output_size: int = embedding_dim
                # Generate the layer mode
                layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                    layer_idx=layer_idx,
                    bias_type=bias_type,
                    norm_type=NormType.NONE.value,
                    shortcut_type=ShortcutType.NONE.value,
                    output_size=output_size,
                    input_size=embedding_dim,
                    activation_type=ActivationType.NONE.value,
                    activation_param=0.0,
                    dropout_rate=0.0,
                    input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                    initialization_type=InitializationType.DEFAULT.value
                )
                # Update the embedding dimension
                embedding_dim = output_size
        else:
            # Output layer
            output_size: int = int(param_specified["output_size"])
            # Generate the layer mode
            layer_mode: Dict[str, float] = LinearLayerUtils.get_layer_mode(
                layer_idx=layer_idx,
                bias_type=BiasType.WITH_BIAS.value,
                norm_type=NormType.NONE.value,
                shortcut_type=ShortcutType.NONE.value,
                output_size=output_size,
                input_size=embedding_dim,
                activation_type=ActivationType.NONE.value,
                activation_param=0.0,
                dropout_rate=0.0,
                input_pooling_reshape_type=InputPoolingReshapeType.NONE.value,
                initialization_type=InitializationType.DEFAULT.value
            )
            # Update the embedding dimension
            embedding_dim = output_size                
        local_mode.append(layer_mode)
    return local_mode

@jit_script
def _retrieve_max_memory_size(param_specified: Dict[str, float],
                             param_sampled: Dict[str, List[float]]
                             ) -> Tuple[int, int, int, int, int]:
    '''
    Retrieve the maximum should-be pre-allocated memory size for encoding,
        shared elements and arange tensor for Transformer model.
    '''
    global_structure_size, local_structure_size, encode_memory_size, \
        shared_element_buffer_size, arange_tensor_size = \
        ModelUtils.retrieve_max_memory_size(param_specified, param_sampled)
    
    # Encoder num
    encoder_num_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                            "encoder_num", "max")
    # Linear layer num in each encoder
    linear_layer_num_per_encoder_max: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                             "linear_layer_num_per_encoder", "max")
    # Embedding dimension
    embedding_dim_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                "embedding_dim", "max")
    
    param_specified_max: Dict[str, float] = {
        "encoder_num": float(encoder_num_max),
        "linear_layer_num_per_encoder": float(linear_layer_num_per_encoder_max),
        "vocab_size": param_specified["vocab_size"],
        "embedding_dim": float(embedding_dim_max),
        "output_size": param_specified["output_size"],
        "max_sequence_length": param_specified["max_sequence_length"],
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
    global_structure_minmax[:, 0] = float(ModelType.TRANSFORMER.value)
    # Task type
    global_structure_minmax[:, 1] = float(param_specified["task_type"])
    # Dataset type
    global_structure_minmax[:, 2] = float(param_specified["dataset_type"])
    # Encoder num
    num_encoder_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                            "encoder_num", "min")
    num_encoder_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                            "encoder_num", "max")
    # Linear layer num in each encoder
    num_linear_layer_per_encoder_min: int = ModelUtils.sample_int_param(param_specified, param_sampled, 
                                                             "linear_layer_num_per_encoder", "min")
    num_linear_layer_per_encoder_max: int = ModelUtils.sample_int_param(param_specified, param_sampled,
                                                             "linear_layer_num_per_encoder", "max")
    # Layer num, embedding layer + encoder layers + output layer
    num_layer_min: int = 1 + num_encoder_min * (num_linear_layer_per_encoder_min + 1) + 1
    num_layer_max: int = 1 + num_encoder_max * (num_linear_layer_per_encoder_max + 1) + 1
    global_structure_minmax[0, 3] = float(num_layer_min)
    global_structure_minmax[1, 3] = float(num_layer_max)
    # Number of encoders
    global_structure_minmax[0, 5] = float(num_encoder_min)
    global_structure_minmax[1, 5] = float(num_encoder_max)
    # --- Local structure minmax ---
    # Layer idx
    local_structure_minmax[0, 0] = 0.0
    local_structure_minmax[1, 0] = float(num_layer_max - 1)
    # Layer type, embedding, multihead attention, linear
    local_structure_minmax[0, 1] = float(LayerType.LINEAR.value)
    local_structure_minmax[1, 1] = float(LayerType.MHA.value)
    # Bias type, always assume the possiblity of bias
    local_structure_minmax[0, 2] = float(BiasType.NONE.value)   
    local_structure_minmax[1, 2] = float(BiasType.WITH_BIAS.value)
    # Norm type, always assume all possibilities of norm
    local_structure_minmax[0, 3] = float(NormType.NONE.value)
    local_structure_minmax[1, 3] = float(NormType.ACTIVATION_NORM.value)
    # Shortcut type, always assume all possibilities of shortcut
    local_structure_minmax[0, 4] = float(ShortcutType.NONE.value)
    local_structure_minmax[1, 4] = float(ShortcutType.STRAIGHT.value)
    # Output / input embedding dim
    model_output_size: int = int(param_specified["output_size"])
    min_embedding_dim: int = ModelUtils.sample_int_param(param_specified, param_sampled,
        "embedding_dim", "min")
    max_embedding_dim: int = ModelUtils.sample_int_param(param_specified, param_sampled,
        "embedding_dim", "max")
    min_output_size: int = min(model_output_size, min_embedding_dim)
    max_output_size: int = max(model_output_size, max_embedding_dim)
    max_input_size: int = max_embedding_dim
    local_structure_minmax[0, 5] = float(min_output_size)
    local_structure_minmax[1, 5] = float(max_output_size)
    local_structure_minmax[0, 6] = 0
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
    # Input pooling or reshape type, always assume all possibilities of pooling or reshape
    local_structure_minmax[0, 10] = float(InputPoolingReshapeType.NONE.value)
    local_structure_minmax[1, 10] = float(InputPoolingReshapeType.FIRST_TOKEN.value)
    # Number of heads
    local_structure_minmax[0, 14] = 0.0
    local_structure_minmax[1, 14] = float(ModelUtils.sample_int_param(param_specified, param_sampled,
        "num_heads", "max"))
    # Embedding num
    local_structure_minmax[0, 16] = 0.0
    local_structure_minmax[1, 16] = param_specified["vocab_size"]
    # Max sequence length
    local_structure_minmax[0, 17] = 0.0
    local_structure_minmax[1, 17] = param_specified["max_sequence_length"]
    # Initialization type, always assume default initialization
    local_structure_minmax[0, 20] = float(InitializationType.DEFAULT.value)
    local_structure_minmax[1, 20] = float(InitializationType.DEFAULT.value)
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
    # Embedding idx, start from -1 to vocab size - 1
    index_encoding_minmax[0, 7] = -1.0
    index_encoding_minmax[1, 7] = param_specified["vocab_size"] - 1
    # Sequence length idx, start from -1 to max sequence length - 1
    index_encoding_minmax[0, 8] = -1.0
    index_encoding_minmax[1, 8] = param_specified["max_sequence_length"] - 1
    
    return global_structure_minmax, local_structure_minmax, index_encoding_minmax

@jit_script
def _apply_weights(x: torch.Tensor, 
                  padding_mask: Optional[torch.Tensor],
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
        if layer_type == LayerType.MHA.value:
            # Apply the MHA layer with padding mask
            x = MultiheadAttentionLayerUtils.apply_params(
                x=x,
                padding_mask=padding_mask,
                layer_structure=local_mode[layer_idx],
                layer_params=layers_params[layer_idx],
                layer_param_shapes=layers_param_shapes[layer_idx],
                training=training
            )
        else:
            # Apply embedding or linear layer
            x = layer_apply_params_1i(
                layer_type=layer_type,
                x=x,
                layer_structure=local_mode[layer_idx],
                layer_params=layers_params[layer_idx],
                layer_param_shapes=layers_param_shapes[layer_idx],
                training=training
            )
    return x

class TransformerModelUtils:
    '''
    Transformer model encoding & forwarding and modularization
    '''
    get_model_mode = staticmethod(_get_model_mode)
    generate_mode = staticmethod(_generate_mode)
    generate_mode_for_size = staticmethod(_generate_mode_for_size)
    retrieve_max_memory_size = staticmethod(_retrieve_max_memory_size)
    retrieve_encode_input_minmax = staticmethod(_retrieve_encode_input_minmax)
    apply_weights = staticmethod(_apply_weights)

class TransformerModel(Model):
    '''
    Transformer model encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        '''
        Forward the input tensor through the GAT model.
        '''
        # Forward the input tensor through the GAT layers
        for layer in self.layers:
            if isinstance(layer, MultiheadAttentionLayer):
                # Apply the MHA layer with padding mask
                x = layer(x, padding_mask)
            else:
                # Apply the embedding or linear layer
                x = layer(x)
        return x
