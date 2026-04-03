'''
Kolmogorov-Arnold Network Layer encoding & forwarding and modularization
'''

from .layer import *

@jit_script
def cumsum(x: torch.Tensor) -> torch.Tensor:
    '''
    Compute the cumulative sum of the input tensor along the last dimension.
    This is a naive implementation, as the grid is small, the performance is acceptable.
    Notice that the torch.Tensor.cumsum is not reproducible.
    '''
    y = torch.zeros_like(x)
    for idx in range(x.shape[-1]):
        if idx == 0:
            y[:, idx] = x[:, idx]
        else:
            y[:, idx] = y[:, idx - 1] + x[:, idx]
    return y

@jit_script
def b_splines(x: torch.Tensor,
              grid_lower_bound: torch.Tensor,
              grid_length: torch.Tensor,
              grid_knots: torch.Tensor,
              spline_order: int
              ) -> torch.Tensor:
    '''
    Compute the B-spline basis functions.
    '''
    grid = grid_lower_bound.unsqueeze(1) + torch.exp(grid_length).unsqueeze(1) * \
        cumsum(torch.softmax(grid_knots, dim=-1))
    # Compute the B-spline basis functions
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x)
            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )
    return bases.view(x.shape[0], -1)

class KANLayerUtils:
    '''
    KAN layer encoding & forwarding and modularization
    '''
    @staticmethod
    @jit_script
    def get_layer_mode(layer_idx: int,
                         bias_type: int,
                         output_size: int,
                         input_size: int,
                         base_activation_type: int,
                         base_activation_param: float,
                         grid_size: int,
                         spline_order: int,
                         initialization_type: int
                         ) -> Dict[str, float]:
        '''
        Get a layer mode dictionary for Kolmogorov-Arnold Network layer.
        '''
        layer_mode: Dict[str, float] = LayerUtils.get_layer_mode()
        layer_mode.update({
            "layer_idx": float(layer_idx),
            "layer_type": float(LayerType.KAN.value),
            "bias_type": float(bias_type),
            "output_size": float(output_size),
            "input_size": float(input_size),
            "activation_type": float(base_activation_type),
            "activation_param": base_activation_param,
            "grid_size": float(grid_size),
            "spline_order": float(spline_order),
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
        For KAN layer, encode param type, output idx, input idx and grid idx
            besides the basic index encoding in the LayerUtils class.
        '''
        LayerUtils.encode_index(layer_structure, shared_element, 
                             shared_memory, unique_memory, arange_tensor)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_size: int = int(layer_structure["output_size"])
        input_size: int = int(layer_structure["input_size"])
        grid_size: int = int(layer_structure["grid_size"])
        spline_order: int = int(layer_structure["spline_order"])

        # Retrieve pre-allocated memory for encoding
        param_type_base_weight, param_type_base_bias, param_type_spline_weight, \
            param_type_spline_scale, param_type_grid_lower_bound, param_type_grid_length, \
            param_type_grid_knot = unique_memory[IndexType.Unique.PARAM_TYPE.name]
        output_idx_base_weight, output_idx_base_bias, output_idx_spline_weight, \
            output_idx_spline_scale, _, _, _ = unique_memory[IndexType.Unique.OUTPUT_IDX.name]
        input_idx_base_weight, _, input_idx_spline_weight, \
            input_idx_spline_scale, input_idx_grid_lower_bound, input_idx_grid_length, \
            input_idx_grid_knot = unique_memory[IndexType.Unique.INPUT_IDX.name]
        _, _, grid_idx_spline_weight, \
            _, _, _, \
            grid_idx_grid_knot = unique_memory[IndexType.Unique.GRID_IDX.name]

        # Assign param type encoding
        param_type_base_weight.fill_(ParamType.KAN.BASE_WEIGHTS.value)
        param_type_spline_weight.fill_(ParamType.KAN.SPLINE_WEIGHTS.value)
        param_type_spline_scale.fill_(ParamType.KAN.SPLINE_SCALES.value)
        param_type_grid_lower_bound.fill_(ParamType.KAN.GRID_LOWER_BOUNDS.value)
        param_type_grid_length.fill_(ParamType.KAN.GRID_LENGTHS.value)
        param_type_grid_knot.fill_(ParamType.KAN.GRID_KNOTS.value)

        # --- Base weights & Spline scales (out x in) ---
        # Reshape for encoding base weights first
        output_idx_base_weight = output_idx_base_weight.view(output_size, input_size)
        input_idx_base_weight = input_idx_base_weight.view(output_size, input_size)
        # Retrieve values by slicing arange tensor
        slice_output_idx = arange_tensor[:output_size]
        slice_input_idx = arange_tensor[:input_size]
        # Copy the values to the pre-allocated memory
        output_idx_base_weight.copy_(slice_output_idx.view(-1, 1))
        input_idx_base_weight.copy_(slice_input_idx.view(1, -1))
        # Reshape back for copying to spline scales
        output_idx_base_weight = output_idx_base_weight.view(-1)
        input_idx_base_weight = input_idx_base_weight.view(-1)
        # Copy the values to the spline scales
        output_idx_spline_scale.copy_(output_idx_base_weight)
        input_idx_spline_scale.copy_(input_idx_base_weight)

        # --- Spline weights (out x in x (grid_size + spline_order)) ---
        # Reshape for encoding spline weights
        output_idx_spline_weight = output_idx_spline_weight.view(output_size, input_size, grid_size + spline_order)
        input_idx_spline_weight = input_idx_spline_weight.view(output_size, input_size, grid_size + spline_order)
        grid_idx_spline_weight = grid_idx_spline_weight.view(output_size, input_size, grid_size + spline_order)
        # Retrieve values by slicing arange tensor
        slice_grid_spline_weights_idx = arange_tensor[:grid_size + spline_order]
        # Copy the values to the pre-allocated memory
        output_idx_spline_weight.copy_(slice_output_idx.view(-1, 1, 1))
        input_idx_spline_weight.copy_(slice_input_idx.view(1, -1, 1))
        grid_idx_spline_weight.copy_(slice_grid_spline_weights_idx.view(1, 1, -1))

        # --- Grid lower bounds & Grid lengths (in) ---
        # Encode grid lower bounds and grid lengths
        input_idx_grid_lower_bound.copy_(slice_input_idx)
        input_idx_grid_length.copy_(slice_input_idx)

        # -- Grid knots (in x (grid_size + 2 * spline_order + 1)) ---
        # Reshape for encoding grid knots
        input_idx_grid_knot = input_idx_grid_knot.view(input_size, grid_size + 2 * spline_order + 1)
        grid_idx_grid_knot = grid_idx_grid_knot.view(input_size, grid_size + 2 * spline_order + 1)
        # Retrieve values by slicing arange tensor
        slice_grid_knots_idx = arange_tensor[:grid_size + 2 * spline_order + 1]
        # Copy the values to the pre-allocated memory
        input_idx_grid_knot.copy_(slice_input_idx.view(-1, 1))
        grid_idx_grid_knot.copy_(slice_grid_knots_idx.view(1, -1))

        # --- Base bias (out)
        # Encode base bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            # Assign param type encoding
            param_type_base_bias.fill_(ParamType.KAN.BASE_BIASES.value)
            # Encoding output idx
            output_idx_base_bias.copy_(slice_output_idx)

    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding this kan layer.
        '''
        output_size: int = int(layer_structure["output_size"])
        input_size: int = int(layer_structure["input_size"])
        grid_size: int = int(layer_structure["grid_size"])
        spline_order: int = int(layer_structure["spline_order"])
        return max(output_size, input_size, grid_size + spline_order, 
                   grid_size + 2 * spline_order + 1,
                   LayerUtils.retrieve_required_arange_size(layer_structure))
    
    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of KAN layer parameters.
        '''
        layer_lens, layer_shapes = LayerUtils.retrieve_shapes(layer_structure)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_size: int = int(layer_structure["output_size"])
        input_size: int = int(layer_structure["input_size"])
        grid_size: int = int(layer_structure["grid_size"])
        spline_order: int = int(layer_structure["spline_order"])

        # Param of base weights
        layer_lens.append(output_size * input_size)
        layer_shapes.append([output_size, input_size])
        # Param of base bias
        bias_num = output_size if bias_type == BiasType.WITH_BIAS.value else 0
        layer_lens.append(bias_num)
        layer_shapes.append([bias_num])
        # Param of spline weights
        layer_lens.append(output_size * input_size * (grid_size + spline_order))
        layer_shapes.append([output_size, input_size, grid_size + spline_order])
        # Param of spline scales
        layer_lens.append(output_size * input_size)
        layer_shapes.append([output_size, input_size])
        # Param of grid lower bounds
        layer_lens.append(input_size)
        layer_shapes.append([input_size])
        # Param of grid lengths
        layer_lens.append(input_size)
        layer_shapes.append([input_size])
        # Param of grid knots
        layer_lens.append(input_size * (grid_size + 2 * spline_order + 1))
        layer_shapes.append([input_size, grid_size + 2 * spline_order + 1])

        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of this kan layer parameters.
        Use Uniform(-1/sqrt(in_features), 1/sqrt(in_features)) for base weights 
            and spline scales; zero initialization for base biases; initialize the 
            spline weights with a small value as described in footnote 2 in the paper, 
            as in Normal(0, 0.1 ^ 2); initialize the grid lower bounds with constant -1,
            grid lengths with log(2), and grid knots with 0.
        '''
        layer_stats = LayerUtils.get_params_initial_statistic(layer_structure)

        # Retrieve the layer structure
        input_size: int = int(layer_structure["input_size"])
        init_type: int = int(layer_structure["initialization_type"])

        if init_type == InitializationType.DEFAULT.value:
            # Base weights initialized with U(-1/sqrt(in_features), 1/sqrt(in_features))
            target_base_weight_mean = 0.0
            target_base_weight_std = 1 / math.sqrt(3 * input_size)
            layer_stats.append((target_base_weight_mean, target_base_weight_std))
            # Zero initialization for base biases
            layer_stats.append((0.0, 0.0))
            # Spline weights initialized with Normal(0, 0.1 ^ 2)
            layer_stats.append((0.0, 0.1))
            # Spline scales initialized with U(-1/sqrt(in_features), 1/sqrt(in_features))
            layer_stats.append((target_base_weight_mean, target_base_weight_std))
            # Grid lower bounds initialized with -1
            layer_stats.append((-1.0, 0.0))
            # Grid lengths initialized with log(2)
            layer_stats.append((math.log(2), 0.0))
            # Grid knots initialized with 0
            layer_stats.append((0.0, 0.0))
        elif init_type == InitializationType.ZERO.value:
            # Zero initialization for base weights
            layer_stats.append((0.0, 0.0))
            # Zero initialization for base biases
            layer_stats.append((0.0, 0.0))
            # Zero initialization for spline weights
            layer_stats.append((0.0, 0.0))
            # Zero initialization for spline scales
            layer_stats.append((0.0, 0.0))
            # Zero initialization for grid lower bounds
            layer_stats.append((0.0, 0.0))
            # Zero initialization for grid lengths
            layer_stats.append((0.0, 0.0))
            # Zero initialization for grid knots
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
                     training: bool                     
              ) -> torch.Tensor:
        '''
        Apply the KAN layer to the input tensor.
        '''
        x = LayerUtils.apply_params(x, layer_structure, layer_params,
                                 layer_param_shapes, training)
        
        # Retrieve the layer structure
        bias_type = int(layer_structure["bias_type"])
        base_activation_type = int(layer_structure["activation_type"])
        base_activation_param = layer_structure["activation_param"]
        spline_order = int(layer_structure["spline_order"])

        # Retrieve the layer parameters
        base_weight, base_bias, spline_weight, spline_scale, grid_lower_bound, grid_length, grid_knots = layer_params
        base_weight_shape, _, spline_weight_shape, spline_scale_shape, _, _, grid_knot_shape = layer_param_shapes
        # Reshape for the layer
        base_weight = base_weight.view(base_weight_shape)
        spline_weight = spline_weight.view(spline_weight_shape)
        spline_scale = spline_scale.view(spline_scale_shape)
        grid_knots = grid_knots.view(grid_knot_shape)
        base_bias = base_bias if bias_type == BiasType.WITH_BIAS.value else None

        # Base activation
        base_output = LayerUtils.activation(x, base_activation_type, base_activation_param)
        # Linear transformation
        base_output = F.linear(base_output, base_weight, base_bias)

        # B-spline activation
        scaled_spline_weight = (spline_weight * spline_scale.unsqueeze(2)).view(spline_weight_shape[0], -1)
        spline_output = b_splines(x, grid_lower_bound, grid_length, grid_knots, spline_order)
        spline_output = F.linear(spline_output, scaled_spline_weight)

        # Combine the base and spline outputs
        output = base_output + spline_output

        return output

class KANLayer(Layer):
    '''
    KAN layer encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the KAN layer.
        '''
        # Call the apply_params function as forward
        return KANLayerUtils.apply_params(x=x, 
                                          layer_structure=self.layer_structure, 
                                          layer_params=list(self.layer_params), 
                                          layer_param_shapes=self.layer_param_shapes, 
                                          training=self.training)

class KANLayerWrapper(nn.Module):
    def __init__(self, kan_layer: KANLayer) -> None:
        super(KANLayerWrapper, self).__init__()
        layer_structure = kan_layer.layer_structure

        # Retrieve the layer structure
        self.bias_type = int(layer_structure["bias_type"])
        self.base_activation_type = int(layer_structure["activation_type"])
        self.base_activation_param = layer_structure["activation_param"]
        self.spline_order = int(layer_structure["spline_order"])

        # Retrieve the layer parameters
        base_weight, base_bias, spline_weight, spline_scale, grid_lower_bound, grid_length, grid_knots = kan_layer.layer_params
        base_weight_shape, _, spline_weight_shape, spline_scale_shape, _, _, grid_knot_shape = kan_layer.layer_param_shapes
        # Reshape for the layer
        base_weight = base_weight.view(base_weight_shape)
        spline_weight = spline_weight.view(spline_weight_shape)
        spline_scale = spline_scale.view(spline_scale_shape)
        grid_knots = grid_knots.view(grid_knot_shape)
        base_bias = base_bias if self.bias_type == BiasType.WITH_BIAS.value else None
        # Wrap as nn.Parameter
        self.base_weight = nn.Parameter(base_weight)
        if base_bias is not None:
            self.base_bias = nn.Parameter(base_bias)
        else:
            self.register_parameter('base_bias', None)
        self.spline_weight = nn.Parameter(spline_weight)
        self.spline_scale = nn.Parameter(spline_scale)
        self.grid_lower_bound = nn.Parameter(grid_lower_bound)
        self.grid_length = nn.Parameter(grid_length)
        self.grid_knots = nn.Parameter(grid_knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base activation
        base_output = LayerUtils.activation(x, self.base_activation_type, self.base_activation_param)
        # Linear transformation
        base_output = F.linear(base_output, self.base_weight, self.base_bias)

        # B-spline activation
        scaled_spline_weight = (self.spline_weight * self.spline_scale.unsqueeze(2)).view(self.spline_weight.shape[0], -1)
        spline_output = b_splines(x, self.grid_lower_bound, self.grid_length, self.grid_knots, self.spline_order)
        spline_output = F.linear(spline_output, scaled_spline_weight)

        # Combine the base and spline outputs
        output = base_output + spline_output

        return output
    