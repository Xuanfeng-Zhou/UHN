'''
Linear layer encoding & forwarding and modularization
'''

from .layer import *

class LinearLayerUtils:
    '''
    Linear layer encoding & forwarding and modularization
    '''
    @staticmethod
    @jit_script
    def get_layer_mode(layer_idx: int,
                         bias_type: int,
                         norm_type: int,
                         shortcut_type: int,
                         output_size: int,
                         input_size: int,
                         activation_type: int,
                         activation_param: float,
                         dropout_rate: float,
                         input_pooling_reshape_type: int,
                         initialization_type: int
                         ) -> Dict[str, float]:
        '''
        Get a layer mode dictionary for linear layer.
        '''
        layer_mode: Dict[str, float] = LayerUtils.get_layer_mode()
        layer_mode.update({
            "layer_idx": float(layer_idx),
            "layer_type": float(LayerType.LINEAR.value),
            "bias_type": float(bias_type),
            "norm_type": float(norm_type),
            "shortcut_type": float(shortcut_type),
            "output_size": float(output_size),
            "input_size": float(input_size),
            "activation_type": float(activation_type),
            "activation_param": activation_param,
            "dropout_rate": dropout_rate,
            "input_pooling_reshape_type": float(input_pooling_reshape_type),
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
        For linear layer, encode param type, output idx and input idx 
            besides the basic index encoding in the LayerUtils class.
        '''
        LayerUtils.encode_index(layer_structure, shared_element, 
                             shared_memory, unique_memory, arange_tensor)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_size: int = int(layer_structure["output_size"])
        input_size: int = int(layer_structure["input_size"])

        # Retrieve pre-allocated memory for encoding
        param_type_weight, param_type_bias = unique_memory[IndexType.Unique.PARAM_TYPE.name]
        output_idx_weight, output_idx_bias = unique_memory[IndexType.Unique.OUTPUT_IDX.name]
        input_idx_weight, _ = unique_memory[IndexType.Unique.INPUT_IDX.name]

        # Assign param type encoding
        param_type_weight.fill_(ParamType.Linear.WEIGHTS.value)
        # Reshape the weights encoding for the layer
        output_idx_weight = output_idx_weight.view(output_size, input_size)
        input_idx_weight = input_idx_weight.view(output_size, input_size)
        # Retrieve values by slicing arange tensor
        slice_output_idx = arange_tensor[:output_size]
        slice_input_idx = arange_tensor[:input_size]
        # Copy the values to the pre-allocated memory
        output_idx_weight.copy_(slice_output_idx.view(-1, 1))
        input_idx_weight.copy_(slice_input_idx.view(1, -1))

        # Encoding bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            # Assign param type encoding
            param_type_bias.fill_(ParamType.Linear.BIASES.value)
            # Encoding output idx and ignore input idx
            output_idx_bias.copy_(slice_output_idx)

    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding this linear layer.
        '''
        output_size: int = int(layer_structure["output_size"])
        input_size: int = int(layer_structure["input_size"])
        return max(output_size, input_size,
                   LayerUtils.retrieve_required_arange_size(layer_structure))

    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of linear layer parameters.
        '''
        layer_lens, layer_shapes = LayerUtils.retrieve_shapes(layer_structure)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_size: int = int(layer_structure["output_size"])
        input_size: int = int(layer_structure["input_size"])

        # Param of weight
        layer_lens.append(output_size * input_size)
        layer_shapes.append([output_size, input_size])
        # Param of bias
        bias_num = output_size if bias_type == BiasType.WITH_BIAS.value else 0
        layer_lens.append(bias_num)
        layer_shapes.append([bias_num])
        
        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of this linear layer parameters.
        Use Uniform(-1/sqrt(in_features), 1/sqrt(in_features)) for weights,
            and zero initialization for biases.
        '''
        layer_stats = LayerUtils.get_params_initial_statistic(layer_structure)

        # Retrieve the layer structure
        input_size: int = int(layer_structure["input_size"])
        init_type: int = int(layer_structure["initialization_type"])

        if init_type == InitializationType.DEFAULT.value:
            # Weights initialized with Uniform(-1/sqrt(in_features), 1/sqrt(in_features))
            target_weight_mean = 0.0
            target_weight_std = 1 / math.sqrt(3 * input_size)
            layer_stats.append((target_weight_mean, target_weight_std))
            # Use Pytorch's default initialization for biases, equivalent to 
                # bound = 1 / math.sqrt(fan_in), init.uniform_(self.bias, -bound, bound)
            layer_stats.append((target_weight_mean, target_weight_std))
        elif init_type == InitializationType.ZERO.value:
            # Zero initialization for weights
            layer_stats.append((0.0, 0.0))
            # Zero initialization for biases
            layer_stats.append((0.0, 0.0))
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}")

        return layer_stats

    @staticmethod
    @jit_script
    def apply_params(x: torch.Tensor,
                     layer_structure: Dict[str, float],
                     layer_params: List[torch.Tensor],
                     layer_param_shapes: List[List[int]],
                     training: bool,
                     ) -> torch.Tensor:
        '''
        Apply the linear layer to the input tensor.
        This function is based on Pre-activation residual connection.
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
        pooling_type: int = int(layer_structure["input_pooling_reshape_type"])

        # Retrieve the layer parameters
        weight, bias = layer_params
        weight_shape, _ = layer_param_shapes
        # Reshape for the layer
        weight = weight.view(weight_shape)
        bias = bias if bias_type == BiasType.WITH_BIAS.value else None

        # Apply straight through, pooling or reshape to the input tensor
        batch_size: int = x.shape[0]
        if pooling_type == InputPoolingReshapeType.AVG_POOLING.value:
            # Avg pooling for 4d input then reshape to (batch_size, dim)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        elif pooling_type == InputPoolingReshapeType.MAX_POOLING.value:
            # Max pooling for 4d input then reshape to (batch_size, dim)
            x = F.adaptive_max_pool2d(x, (1, 1)).view(batch_size, -1)
        elif pooling_type == InputPoolingReshapeType.RESHAPE_TO_2D.value:
            # Reshape to (batch_size, dim)
            x = x.view(batch_size, -1)
        elif pooling_type == InputPoolingReshapeType.AVG_SEQUENCE.value:
            # Avg pooling for (batch_size, seq_len, dim) input to (batch_size, dim)
            x = x.mean(dim=1)
        elif pooling_type == InputPoolingReshapeType.FIRST_TOKEN.value:
            # Use the first token for (batch_size, seq_len, dim) input to (batch_size, dim)
            x = x[:, 0, :]

        # Restore input for shortcut
        shortcut = x

        # Activation
        x = LayerUtils.activation(x, activation_type, activation_param)

        # Normalization
        if norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (x.shape[-1],))

        # Linear transformation
        x = F.linear(x, weight, bias)

        # Dropout
        x = LayerUtils.dropout(x, dropout_rate, training)

        # Shortcut connection
        if shortcut_type == ShortcutType.STRAIGHT.value:
            x = x + shortcut

        return x

class LinearLayer(Layer):
    '''
    Linear layer encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the linear layer.
        '''
        # Call the apply_params function as forward
        return LinearLayerUtils.apply_params(x=x, 
                                             layer_structure=self.layer_structure, 
                                             layer_params=list(self.layer_params), 
                                             layer_param_shapes=self.layer_param_shapes, 
                                             training=self.training)

class LinearLayerWrapper(nn.Module):
    def __init__(self, linear_layer: LinearLayer) -> None:
        super(LinearLayerWrapper, self).__init__()
        layer_structure = linear_layer.layer_structure

        # Retrieve the layer structure
        self.bias_type: int = int(layer_structure["bias_type"])
        self.norm_type: int = int(layer_structure["norm_type"])
        self.shortcut_type: int = int(layer_structure["shortcut_type"])
        self.activation_type: int = int(layer_structure["activation_type"])
        self.activation_param: float = layer_structure["activation_param"]
        self.dropout_rate: float = layer_structure["dropout_rate"]
        self.pooling_type: int = int(layer_structure["input_pooling_reshape_type"])
        self.input_size: int = int(layer_structure["input_size"])

        # Retrieve the layer parameters
        weight, bias = linear_layer.layer_params
        weight_shape, _ = linear_layer.layer_param_shapes
        # Reshape for the layer
        weight = weight.view(weight_shape)
        bias = bias if self.bias_type == BiasType.WITH_BIAS.value else None
        # Wrap as nn.Parameter
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply straight through, pooling or reshape to the input tensor
        batch_size: int = x.shape[0]
        if self.pooling_type == InputPoolingReshapeType.AVG_POOLING.value:
            # Avg pooling for 4d input then reshape to (batch_size, dim)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        elif self.pooling_type == InputPoolingReshapeType.MAX_POOLING.value:
            # Max pooling for 4d input then reshape to (batch_size, dim)
            x = F.adaptive_max_pool2d(x, (1, 1)).view(batch_size, -1)
        elif self.pooling_type == InputPoolingReshapeType.RESHAPE_TO_2D.value:
            # Reshape to (batch_size, dim)
            x = x.view(batch_size, -1)
        elif self.pooling_type == InputPoolingReshapeType.AVG_SEQUENCE.value:
            # Avg pooling for (batch_size, seq_len, dim) input to (batch_size, dim)
            x = x.mean(dim=1)
        elif self.pooling_type == InputPoolingReshapeType.FIRST_TOKEN.value:
            # Use the first token for (batch_size, seq_len, dim) input to (batch_size, dim)
            x = x[:, 0, :]

        # Restore input for shortcut
        shortcut = x

        # Activation
        x = LayerUtils.activation(x, self.activation_type, self.activation_param)

        # Normalization
        if self.norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (self.input_size,))

        # Linear transformation
        x = F.linear(x, self.weight, self.bias)

        # Dropout
        x = LayerUtils.dropout(x, self.dropout_rate, self.training)

        # Shortcut connection
        if self.shortcut_type == ShortcutType.STRAIGHT.value:
            x = x + shortcut

        return x
