'''
Convolutional layer
'''

from .layer import *

class ConvLayerUtils:
    '''
    Convolutional layer encoding & forwarding and modularization
    '''
    @staticmethod
    @jit_script
    def get_layer_mode(layer_idx: int,
                         bias_type: int,
                         norm_type: int,
                         shortcut_type: int,
                         output_channel_dim: int,
                         input_channel_dim: int,
                         activation_type: int,
                         activation_param: float,
                         dropout_rate: float,
                         group_num: int,
                         kernel_size: int,
                         stage_wise_pooling_type: int,
                         initialization_type: int,
                         ) -> Dict[str, float]:
        '''
        Get a layer mode dictionary for convolution layer.
        '''
        layer_mode: Dict[str, float] = LayerUtils.get_layer_mode()
        layer_mode.update({
            "layer_idx": float(layer_idx),
            "layer_type": float(LayerType.CONV.value),
            "bias_type": float(bias_type),
            "norm_type": float(norm_type),
            "shortcut_type": float(shortcut_type),
            "output_size": float(output_channel_dim),
            "input_size": float(input_channel_dim),
            "activation_type": float(activation_type),
            "activation_param": activation_param,
            "dropout_rate": dropout_rate,
            "group_num": float(group_num),
            "kernel_size": float(kernel_size),
            "stage_wise_pooling_type": float(stage_wise_pooling_type),
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
        For convolutional layer, encode param type, output idx, input idx, kernel h idx and kernel w idx
            besides the basic index encoding in the LayerUtils class.
        '''
        LayerUtils.encode_index(layer_structure, shared_element, 
                             shared_memory, unique_memory, arange_tensor)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_channel_dim: int = int(layer_structure["output_size"])
        input_channel_dim: int = int(layer_structure["input_size"])
        kernel_size: int = int(layer_structure["kernel_size"])

        # Retrieve pre-allocated memory for encoding
        param_type_weight, param_type_bias = unique_memory[IndexType.Unique.PARAM_TYPE.name]
        output_idx_weight, output_idx_bias = unique_memory[IndexType.Unique.OUTPUT_IDX.name]
        input_idx_weight, _ = unique_memory[IndexType.Unique.INPUT_IDX.name]
        kernel_h_idx_weight, _ = unique_memory[IndexType.Unique.KERNEL_H_IDX.name]
        kernel_w_idx_weight, _ = unique_memory[IndexType.Unique.KERNEL_W_IDX.name]

        # Assign param type encoding
        param_type_weight.fill_(ParamType.Conv.WEIGHTS.value)
        # Reshape the weights encoding for the layer
        output_idx_weight = output_idx_weight.view(output_channel_dim, input_channel_dim, kernel_size, kernel_size)
        input_idx_weight = input_idx_weight.view(output_channel_dim, input_channel_dim, kernel_size, kernel_size)
        kernel_h_idx_weight = kernel_h_idx_weight.view(output_channel_dim, input_channel_dim, kernel_size, kernel_size)
        kernel_w_idx_weight = kernel_w_idx_weight.view(output_channel_dim, input_channel_dim, kernel_size, kernel_size)
        # Retrieve values by slicing arange tensor
        slice_output_idx = arange_tensor[:output_channel_dim]
        slice_input_idx = arange_tensor[:input_channel_dim]
        slice_kernel_idx = arange_tensor[:kernel_size]
        # Copy the values to the pre-allocated memory
        output_idx_weight.copy_(slice_output_idx.view(-1, 1, 1, 1))
        input_idx_weight.copy_(slice_input_idx.view(1, -1, 1, 1))
        kernel_h_idx_weight.copy_(slice_kernel_idx.view(1, 1, -1, 1))
        kernel_w_idx_weight.copy_(slice_kernel_idx.view(1, 1, 1, -1))

        # Encode bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            # Assign param type encoding
            param_type_bias.fill_(ParamType.Conv.BIASES.value)
            # Encoding output idx and ignore input idx, kernel h idx and kernel w idx
            output_idx_bias.copy_(slice_output_idx)

    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding this convolution layer.
        '''
        output_channel_dim: int = int(layer_structure["output_size"])
        input_channel_dim: int = int(layer_structure["input_size"])
        kernel_size: int = int(layer_structure["kernel_size"])
        return max(output_channel_dim, input_channel_dim, kernel_size,
                   LayerUtils.retrieve_required_arange_size(layer_structure))

    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of convolutional layer parameters.
        '''
        layer_lens, layer_shapes = LayerUtils.retrieve_shapes(layer_structure)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_channel_dim: int = int(layer_structure["output_size"])
        input_channel_dim: int = int(layer_structure["input_size"])
        kernel_size: int = int(layer_structure["kernel_size"])        

        # Param of weight
        weight_num = output_channel_dim * input_channel_dim * kernel_size * kernel_size
        # Param of bias
        if bias_type == BiasType.WITH_BIAS.value:
            bias_num = output_channel_dim
            bias_shape = [output_channel_dim]
        else:
            bias_num = 0
            bias_shape = [0]

        # ----------- Weight ------------
        layer_lens.append(weight_num)
        layer_shapes.append([output_channel_dim, input_channel_dim, kernel_size, kernel_size])
        # ----------- Bias ------------
        layer_lens.append(bias_num)
        layer_shapes.append(bias_shape)

        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of this convolution layer parameters.
        Use Uniform(-1/sqrt(in_features), 1/sqrt(in_features)) for weights,
            and zero initialization for biases.
        '''
        layer_stats = LayerUtils.get_params_initial_statistic(layer_structure)

        # Retrieve the layer structure
        input_channel_dim: int = int(layer_structure["input_size"])
        kernel_size: int = int(layer_structure["kernel_size"])   
        init_type: int = int(layer_structure["initialization_type"])

        if init_type == InitializationType.DEFAULT.value:
            # Calculate the fan_in
            fan_in = input_channel_dim * kernel_size * kernel_size
            # Weights initialized with Uniform(-1/sqrt(in_features), 1/sqrt(in_features))
            target_weight_mean = 0.0
            target_weight_std = 1 / math.sqrt(3 * fan_in)
            layer_stats.append((target_weight_mean, target_weight_std))
            # Use Pytorch's default initialization for biases, equivalent to 
                # bound = 1 / math.sqrt(fan_in), init.uniform_(self.bias, -bound, bound)
            layer_stats.append((target_weight_mean, target_weight_std))
        elif init_type == InitializationType.ZERO.value:
            # Weights initialized with zero
            layer_stats.append((0.0, 0.0))
            # Biases initialized with zero
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
        Apply the convolutional layer to the input tensor.
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
        group_num: int = int(layer_structure["group_num"])
        kernel_size: int = int(layer_structure["kernel_size"])
        stage_wise_pooling_type: int = int(layer_structure["stage_wise_pooling_type"])

        # Retrieve the layer parameters
        weight, bias = layer_params
        weight_shape, _ = layer_param_shapes
        # Reshape for the layer
        weight = weight.view(weight_shape)
        bias = bias if bias_type == BiasType.WITH_BIAS.value else None

        # Restore input for shortcut
        shortcut = x

        # Activation
        x = LayerUtils.activation(x, activation_type, activation_param)

        # Normalization
        if norm_type == NormType.ACTIVATION_NORM.value:
            x = F.group_norm(x, group_num)

        # Pooling
        if stage_wise_pooling_type == StageWisePoolingType.AVG_POOLING.value:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        elif stage_wise_pooling_type == StageWisePoolingType.MAX_POOLING.value:
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Check convolution transition between stages
        conv_stride = 2 if stage_wise_pooling_type == StageWisePoolingType.CONV_POOLING.value else 1
        # Apply convolution
        x = F.conv2d(x, weight, bias, stride=conv_stride, padding=kernel_size // 2)

        # Dropout
        x = LayerUtils.dropout(x, dropout_rate, training)

        # Shortcut connection
        if shortcut_type == ShortcutType.STRAIGHT.value:
            x = x + shortcut
        
        return x

class ConvLayer(Layer):
    '''
    Convolutional layer encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward function for the convolutional layer.
        '''
        # Call the apply_params function as forward
        return ConvLayerUtils.apply_params(x=x, 
                                           layer_structure=self.layer_structure, 
                                           layer_params=list(self.layer_params), 
                                           layer_param_shapes=self.layer_param_shapes, 
                                           training=self.training)

class ConvLayerWrapper(nn.Module):
    '''
    A wrapper class for ConvLayer for the convenience of ONNX export.
    '''
    def __init__(self, conv_layer: ConvLayer) -> None:
        super(ConvLayerWrapper, self).__init__()
        layer_structure = conv_layer.layer_structure

        # Retrieve the layer structure
        self.bias_type: int = int(layer_structure["bias_type"])
        self.norm_type: int = int(layer_structure["norm_type"])
        self.shortcut_type: int = int(layer_structure["shortcut_type"])
        self.activation_type: int = int(layer_structure["activation_type"])
        self.activation_param: float = layer_structure["activation_param"]
        self.dropout_rate: float = layer_structure["dropout_rate"]
        self.group_num: int = int(layer_structure["group_num"])
        self.kernel_size: int = int(layer_structure["kernel_size"])
        self.stage_wise_pooling_type: int = int(layer_structure["stage_wise_pooling_type"])

        # Retrieve the layer parameters
        weight, bias = conv_layer.layer_params
        weight_shape, _ = conv_layer.layer_param_shapes
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
        '''
        Forward function for the convolutional layer.
        '''
        # Restore input for shortcut
        shortcut = x

        # Activation
        x = LayerUtils.activation(x, self.activation_type, self.activation_param)

        # Normalization
        if self.norm_type == NormType.ACTIVATION_NORM.value:
            x = F.group_norm(x, self.group_num)

        # Pooling
        if self.stage_wise_pooling_type == StageWisePoolingType.AVG_POOLING.value:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        elif self.stage_wise_pooling_type == StageWisePoolingType.MAX_POOLING.value:
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Check convolution transition between stages
        conv_stride = 2 if self.stage_wise_pooling_type == StageWisePoolingType.CONV_POOLING.value else 1
        # Apply convolution
        x = F.conv2d(x, self.weight, self.bias, stride=conv_stride, padding=self.kernel_size // 2)

        # Dropout
        x = LayerUtils.dropout(x, self.dropout_rate, self.training)

        # Shortcut connection
        if self.shortcut_type == ShortcutType.STRAIGHT.value:
            x = x + shortcut
        
        return x
