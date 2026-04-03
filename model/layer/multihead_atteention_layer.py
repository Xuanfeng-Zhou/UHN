'''
Multihead Attention Layer
'''

from .layer import *

class MultiheadAttentionLayerUtils:
    '''
    Multihead Attention layer encoding & forwarding and modularization
    '''
    @staticmethod
    @jit_script
    def get_layer_mode(layer_idx: int,
                         bias_type: int,
                         norm_type: int,
                         shortcut_type: int,
                         embedding_dim: int,
                         activation_type: int,
                         activation_param: float,
                         dropout_rate: float,
                         num_heads: int,
                         initialization_type: int
                         ) -> Dict[str, float]:
        '''
        Get a layer mode dictionary for multihead attention layer.
        '''
        layer_mode: Dict[str, float] = LayerUtils.get_layer_mode()
        layer_mode.update({
            "layer_idx": float(layer_idx),
            "layer_type": float(LayerType.MHA.value),
            "bias_type": float(bias_type),
            "norm_type": float(norm_type),
            "shortcut_type": float(shortcut_type),
            "output_size": float(embedding_dim),
            "input_size": float(embedding_dim),
            "activation_type": float(activation_type),
            "activation_param": activation_param,
            "dropout_rate": dropout_rate,
            "num_heads": float(num_heads),
            "initialization_type": float(initialization_type)
        })
        return layer_mode

    @staticmethod
    @jit_script
    def encode_index(layer_structure: Dict[str, float],
                     shared_element: torch.Tensor,
                     shared_memory: torch.Tensor,
                     unique_memory: Dict[str, List[torch.Tensor]],
                     arange_tensor: torch.Tensor
                     ) -> None:
        '''
        Encode the index of the layer to a preallocated memory.
        For multihead attention layer, encode param type, output idx and input idx 
            besides the basic index encoding in the LayerUtils class.
        '''
        LayerUtils.encode_index(layer_structure, shared_element, 
                             shared_memory, unique_memory, arange_tensor)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        embedding_dim: int = int(layer_structure["output_size"])
        
        # Retrieve pre-allocated memory for encoding
        param_type_q_weight, param_type_k_weight, param_type_v_weight, \
            param_type_q_bias, param_type_k_bias, param_type_v_bias, \
            param_type_outproj_weight, param_type_outproj_bias = unique_memory[IndexType.Unique.PARAM_TYPE.name]
        output_idx_q_weight, output_idx_k_weight, output_idx_v_weight, \
            output_idx_q_bias, output_idx_k_bias, output_idx_v_bias, \
            output_idx_outproj_weight, output_idx_outproj_bias = unique_memory[IndexType.Unique.OUTPUT_IDX.name]
        input_idx_q_weight, input_idx_k_weight, input_idx_v_weight, \
            _, _, _, \
            input_idx_outproj_weight, _ = unique_memory[IndexType.Unique.INPUT_IDX.name]
        
        # Assign param type encoding
        param_type_q_weight.fill_(ParamType.MHA.PROJECTION_WEIGHTS_Q.value)
        param_type_k_weight.fill_(ParamType.MHA.PROJECTION_WEIGHTS_K.value)
        param_type_v_weight.fill_(ParamType.MHA.PROJECTION_WEIGHTS_V.value)
        param_type_outproj_weight.fill_(ParamType.MHA.OUT_PROJECTION_WEIGHTS.value)
        # Reshape the weights encoding for the layer
        output_idx_q_weight = output_idx_q_weight.view(embedding_dim, embedding_dim)
        input_idx_q_weight = input_idx_q_weight.view(embedding_dim, embedding_dim)
        # Retrieve values by slicing arange tensor
        slice_idx = arange_tensor[:embedding_dim]
        # Copy the values to the pre-allocated memory
        output_idx_q_weight.copy_(slice_idx.view(-1, 1))
        input_idx_q_weight.copy_(slice_idx.view(1, -1))
        # Reshape back for copying
        output_idx_q_weight = output_idx_q_weight.view(-1)
        input_idx_q_weight = input_idx_q_weight.view(-1)
        # Copy the values to the rest of the weights
        output_idx_k_weight.copy_(output_idx_q_weight)
        output_idx_v_weight.copy_(output_idx_q_weight)
        output_idx_outproj_weight.copy_(output_idx_q_weight)
        input_idx_k_weight.copy_(input_idx_q_weight)
        input_idx_v_weight.copy_(input_idx_q_weight)
        input_idx_outproj_weight.copy_(input_idx_q_weight)

        # Encode bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            # Assign param type encoding
            param_type_q_bias.fill_(ParamType.MHA.PROJECTION_BIASES_Q.value)
            param_type_k_bias.fill_(ParamType.MHA.PROJECTION_BIASES_K.value)
            param_type_v_bias.fill_(ParamType.MHA.PROJECTION_BIASES_V.value)
            param_type_outproj_bias.fill_(ParamType.MHA.OUT_PROJECTION_BIASES.value)
            # Encoding output idx and ignore input idx
            output_idx_q_bias.copy_(slice_idx)
            output_idx_k_bias.copy_(slice_idx)
            output_idx_v_bias.copy_(slice_idx)
            output_idx_outproj_bias.copy_(slice_idx)
            
    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding this linear layer.
        '''
        embedding_dim: int = int(layer_structure["output_size"])
        return max(embedding_dim,
                   LayerUtils.retrieve_required_arange_size(layer_structure))
    
    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of multihead attention layer parameters.
        '''
        layer_lens, layer_shapes = LayerUtils.retrieve_shapes(layer_structure)
        
        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        embedding_dim: int = int(layer_structure["output_size"])

        # Param of weight
        weight_num = embedding_dim * embedding_dim
        # Param of bias
        bias_num = embedding_dim if bias_type == BiasType.WITH_BIAS.value else 0

        # QKV weights
        layer_lens += [weight_num] * 3
        layer_shapes += [[embedding_dim, embedding_dim]] * 3
        # QKV biases
        layer_lens += [bias_num] * 3
        layer_shapes += [[bias_num]] * 3
        # Output projection weights
        layer_lens += [weight_num]
        layer_shapes += [[embedding_dim, embedding_dim]]
        # Output projection biases
        layer_lens += [bias_num]
        layer_shapes += [[bias_num]]

        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of this mha layer parameters.
        Use Xavier initialization for QKV weights, Uniform(-1/sqrt(in_features), 
            1/sqrt(in_features)) for outproj weights and zero initialization for 
            all biases.
        '''
        layer_stats = LayerUtils.get_params_initial_statistic(layer_structure)

        # Retrieve the layer structure
        embedding_dim: int = int(layer_structure["output_size"])
        init_type: int = int(layer_structure["initialization_type"])

        if init_type == InitializationType.DEFAULT.value:
            # QKV weights initialization with Xavier, notice that the fan_in should be
            #   the same as the embedding_dim, as well as the fan_out
            target_qkv_weight_mean = 0.0
            target_qkv_weight_std = math.sqrt(2.0 / (embedding_dim + embedding_dim))
            layer_stats += [(target_qkv_weight_mean, target_qkv_weight_std)] * 3
            # Zero initialization for QKV biases
            layer_stats += [(0.0, 0.0)] * 3
            # Output projection weights initialization with U(-1/sqrt(in_features), 1/sqrt(in_features)) 
            target_outproj_weight_mean = 0.0
            target_outproj_weight_std = 1 / math.sqrt(3 * embedding_dim)
            layer_stats.append((target_outproj_weight_mean, target_outproj_weight_std))
            # Zero initialization for output projection biases
            layer_stats.append((0.0, 0.0))
        elif init_type == InitializationType.ZERO.value:
            # Zero initialization for QKV weights
            layer_stats += [(0.0, 0.0)] * 3
            # Zero initialization for QKV biases
            layer_stats += [(0.0, 0.0)] * 3
            # Zero initialization for output projection weights
            layer_stats.append((0.0, 0.0))
            # Zero initialization for output projection biases
            layer_stats.append((0.0, 0.0))
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}")

        return layer_stats

    @staticmethod
    @jit_script
    def apply_params(x: torch.Tensor,
                     padding_mask: Optional[torch.Tensor],
                     layer_structure: Dict[str, float],
                     layer_params: List[torch.Tensor],
                     layer_param_shapes: List[List[int]],
                     training: bool,                     
                     ) -> torch.Tensor:
        '''
        Apply the multihead attention layer to the input tensor.
        This function is based on Pre-activation residual connection.
        Default as batch first.
        '''
        x = LayerUtils.apply_params(x, layer_structure, layer_params,
                                 layer_param_shapes, training)
        
        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        norm_type: int = int(layer_structure["norm_type"])
        shortcut_type: int = int(layer_structure["shortcut_type"])
        activation_type: int = int(layer_structure["activation_type"])
        activation_param: float = layer_structure["activation_param"]
        dropout_rate: float = layer_structure["dropout_rate"]
        num_heads: int = int(layer_structure["num_heads"])

        # Retrieve the layer parameters
        q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, outproj_weight, outproj_bias = layer_params        
        q_weight_shape, _, _, _, _, _, outproj_weight_shape, _ = layer_param_shapes
        # Reshape for the layer
        in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).view(3 * q_weight_shape[0], q_weight_shape[1])
        outproj_weight = outproj_weight.view(outproj_weight_shape)
        in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0) if bias_type == BiasType.WITH_BIAS.value else None
        outproj_bias = outproj_bias if bias_type == BiasType.WITH_BIAS.value else None
        
        # Restore input for shortcut
        shortcut = x

        # Activation
        x = LayerUtils.activation(x, activation_type, activation_param)

        # Normalization
        if norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (x.shape[-1],))

        # Get float mask
        if padding_mask is not None:
            padding_mask = F._canonical_mask(
                mask=padding_mask,
                mask_name="key_padding_mask",
                other_type=None,
                other_name="attn_mask",
                target_type=x.dtype
            )

        # Transpose for multihead attention
        x_transpose = x.transpose(1, 0)

        # Multihead attention
        x_transpose, _ = F.multi_head_attention_forward(
            query=x_transpose,
            key=x_transpose,
            value=x_transpose,
            embed_dim_to_check=x_transpose.shape[-1],
            num_heads=num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=dropout_rate,
            out_proj_weight=outproj_weight,
            out_proj_bias=outproj_bias,
            training=training,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        # Transpose back to original shape
        x = x_transpose.transpose(1, 0)

        # Dropout
        x = LayerUtils.dropout(x, dropout_rate, training)

        # Residual connection
        if shortcut_type == ShortcutType.STRAIGHT.value:
            x = x + shortcut

        return x

class MultiheadAttentionLayer(Layer):
    '''
    Multihead Attention layer encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass of the linear layer.
        '''
        # Call the apply_params function as forward
        return MultiheadAttentionLayerUtils.apply_params(x=x, 
                                                         padding_mask=padding_mask, 
                                                         layer_structure=self.layer_structure, 
                                                         layer_params=list(self.layer_params), 
                                                         layer_param_shapes=self.layer_param_shapes, 
                                                         training=self.training)

class MultiheadAttentionLayerWrapper(nn.Module):
    def __init__(self, multihead_attention_layer: MultiheadAttentionLayer) -> None:
        super(MultiheadAttentionLayerWrapper, self).__init__()
        layer_structure = multihead_attention_layer.layer_structure

        # Retrieve the layer structure
        self.bias_type: int = int(layer_structure["bias_type"])
        self.norm_type: int = int(layer_structure["norm_type"])
        self.shortcut_type: int = int(layer_structure["shortcut_type"])
        self.activation_type: int = int(layer_structure["activation_type"])
        self.activation_param: float = layer_structure["activation_param"]
        self.dropout_rate: float = layer_structure["dropout_rate"]
        self.num_heads: int = int(layer_structure["num_heads"])
        self.embedding_dim: int = int(layer_structure["output_size"])

        # Retrieve the layer parameters
        q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, outproj_weight, outproj_bias = multihead_attention_layer.layer_params
        q_weight_shape, _, _, _, _, _, outproj_weight_shape, _ = multihead_attention_layer.layer_param_shapes
        # Reshape for the layer
        in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).view(3 * q_weight_shape[0], q_weight_shape[1])
        outproj_weight = outproj_weight.view(outproj_weight_shape)
        in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0) if self.bias_type == BiasType.WITH_BIAS.value else None
        outproj_bias = outproj_bias if self.bias_type == BiasType.WITH_BIAS.value else None
        # Wrap as nn.Parameter
        self.in_proj_weight = nn.Parameter(in_proj_weight)
        self.outproj_weight = nn.Parameter(outproj_weight)
        if self.bias_type == BiasType.WITH_BIAS.value:
            self.in_proj_bias = nn.Parameter(in_proj_bias)
            self.outproj_bias = nn.Parameter(outproj_bias)
        else:
            self.register_parameter('in_proj_bias', None)
            self.register_parameter('outproj_bias', None)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # Restore input for shortcut
        shortcut = x

        # Activation
        x = LayerUtils.activation(x, self.activation_type, self.activation_param)

        # Normalization
        if self.norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (self.embedding_dim,))

        # Get float mask
        if padding_mask is not None:
            padding_mask = F._canonical_mask(
                mask=padding_mask,
                mask_name="key_padding_mask",
                other_type=None,
                other_name="attn_mask",
                target_type=x.dtype
            )

        # Transpose for multihead attention
        x_transpose = x.transpose(1, 0)

        # Multihead attention
        x_transpose, _ = F.multi_head_attention_forward(
            query=x_transpose,
            key=x_transpose,
            value=x_transpose,
            embed_dim_to_check=x_transpose.shape[-1],
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout_rate,
            out_proj_weight=self.outproj_weight,
            out_proj_bias=self.outproj_bias,
            training=self.training,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        # Transpose back to original shape
        x = x_transpose.transpose(1, 0)

        # Dropout
        x = LayerUtils.dropout(x, self.dropout_rate, self.training)

        # Residual connection
        if self.shortcut_type == ShortcutType.STRAIGHT.value:
            x = x + shortcut

        return x
    