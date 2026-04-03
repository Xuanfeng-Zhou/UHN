'''
GCN layer
'''

from .layer import *
from torch_geometric.utils import scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GCNLayerUtils:
    '''
    GCN layer encoding & forwarding and modularization
    '''
    @staticmethod
    @jit_script
    def get_layer_mode(layer_idx: int,
                         bias_type: int,
                         norm_type: int,
                         output_dim: int,
                         input_dim: int,
                         activation_type: int,
                         activation_param: float,
                         dropout_rate: float,
                         initialization_type: int
                         ) -> Dict[str, float]:
        '''
        Get a layer mode dictionary for graph convolution layer.
        '''
        layer_mode: Dict[str, float] = LayerUtils.get_layer_mode()
        layer_mode.update({
            "layer_idx": float(layer_idx),
            "layer_type": float(LayerType.GCN.value),
            "bias_type": float(bias_type),
            "norm_type": float(norm_type),
            "output_size": float(output_dim),
            "input_size": float(input_dim),
            "activation_type": float(activation_type),
            "activation_param": activation_param,
            "dropout_rate": dropout_rate,
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
        For GCN layer, encode param type, output idx and input idx 
            besides the basic index encoding in the LayerUtils class.
        '''
        LayerUtils.encode_index(layer_structure, shared_element, 
                             shared_memory, unique_memory, arange_tensor)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])

        # Retrieve pre-allocated memory for encoding
        param_type_weight, param_type_bias = unique_memory[IndexType.Unique.PARAM_TYPE.name]
        output_idx_weight, output_idx_bias = unique_memory[IndexType.Unique.OUTPUT_IDX.name]
        input_idx_weight, _ = unique_memory[IndexType.Unique.INPUT_IDX.name]

        # Assign param type encoding
        param_type_weight.fill_(ParamType.GCN.WEIGHTS.value)
        # Reshape the weights encoding for the layer
        output_idx_weight = output_idx_weight.view(output_dim, input_dim)
        input_idx_weight = input_idx_weight.view(output_dim, input_dim)
        # Retrieve values by slicing arange tensor
        slice_output_idx = arange_tensor[:output_dim]
        slice_input_idx = arange_tensor[:input_dim]
        # Copy the values to the pre-allocated memory
        output_idx_weight.copy_(slice_output_idx.view(-1, 1))
        input_idx_weight.copy_(slice_input_idx.view(1, -1))

        # Encoding bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            # Assign param type encoding
            param_type_bias.fill_(ParamType.GCN.BIASES.value)
            # Encoding output idx and ignore input idx
            output_idx_bias.copy_(slice_output_idx)

    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding this gcn layer.
        '''
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])
        return max(output_dim, input_dim,
                   LayerUtils.retrieve_required_arange_size(layer_structure))
    
    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of GCN layer parameters.
        '''
        layer_lens, layer_shapes = LayerUtils.retrieve_shapes(layer_structure)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])

        # Param of weight
        layer_lens.append(output_dim * input_dim)
        layer_shapes.append([output_dim, input_dim])
        # Param of bias
        bias_num = output_dim if bias_type == BiasType.WITH_BIAS.value else 0
        layer_lens.append(bias_num)
        layer_shapes.append([bias_num])
        
        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of this gcn layer parameters.
        Use Glort initialization for weights and zero initialization for biases.
        '''
        layer_stats = LayerUtils.get_params_initial_statistic(layer_structure)

        # Retrieve the layer structure
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])
        init_type: int = int(layer_structure["initialization_type"])

        if init_type == InitializationType.DEFAULT.value:
            # Calculate the target mean and std for Glort initialization of weights
            target_weight_mean = 0.0
            target_weight_std = math.sqrt(2.0 / (input_dim + output_dim))
            layer_stats.append((target_weight_mean, target_weight_std))
            # Zero initialization for biases
            layer_stats.append((0.0, 0.0))
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
                     edge_index: torch.Tensor,
                     layer_structure: Dict[str, float],
                     layer_params: List[torch.Tensor],
                     layer_param_shapes: List[List[int]],
                     training: bool,
              ) -> torch.Tensor:
        '''
        Apply the GCN layer to the input tensor.
        Following activation -> dropout -> graph convolution.
        '''
        x = LayerUtils.apply_params(x, layer_structure, layer_params,
                                 layer_param_shapes, training)
        
        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        norm_type: int = int(layer_structure["norm_type"])
        activation_type: int = int(layer_structure["activation_type"])
        activation_param: float = layer_structure["activation_param"]
        dropout_rate: float = layer_structure["dropout_rate"]

        # Retrieve the parameters
        weight, bias = layer_params
        weight_shape, _ = layer_param_shapes
        # Reshape for the layer
        weight = weight.view(weight_shape)

        # Activation
        x = LayerUtils.activation(x, activation_type, activation_param)

        # Layer normalization
        if norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (x.shape[-1],))

        # Dropout
        x = LayerUtils.dropout(x, dropout_rate, training)

        # Linear transformation
        x = F.linear(x, weight)

        # Normalize edge weights
        edge_index, edge_weight = gcn_norm(edge_index,
                                           add_self_loops=True,
                                           num_nodes=x.size(0), 
                                           dtype=x.dtype)
        # Edge weight check for jit scripting, which should never be None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1),
                                     dtype=x.dtype, 
                                     device='cuda')

        # Weighting by edge weights
        # Perform sparse matrix multiplication (A @ X)
        row, col = edge_index[0], edge_index[1] # Source nodes & Target nodes

        x_source_target = x[row] * edge_weight.unsqueeze(1)
        out = scatter(x_source_target, col, dim=0, dim_size=x.size(0), reduce='sum')

        # Add bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            out = out + bias

        return out

class GCNLayer(Layer):
    '''
    GCN layer encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        '''
        Forward the input tensor through the GCN layer.
        '''
        # Call the apply_params function as forward
        return GCNLayerUtils.apply_params(x=x, 
                                          edge_index=edge_index,
                                          layer_structure=self.layer_structure, 
                                          layer_params=list(self.layer_params), 
                                          layer_param_shapes=self.layer_param_shapes, 
                                          training=self.training)

class GCNLayerWrapper(nn.Module):
    def __init__(self, gcn_layer: GCNLayer) -> None:
        super(GCNLayerWrapper, self).__init__()
        layer_structure = gcn_layer.layer_structure

        # Retrieve the layer structure
        self.bias_type: int = int(layer_structure["bias_type"])
        self.norm_type: int = int(layer_structure["norm_type"])
        self.activation_type: int = int(layer_structure["activation_type"])
        self.activation_param: float = layer_structure["activation_param"]
        self.dropout_rate: float = layer_structure["dropout_rate"]
        self.input_size: int = int(layer_structure["input_size"])

        # Retrieve the parameters
        weight, bias = gcn_layer.layer_params
        weight_shape, _ = gcn_layer.layer_param_shapes
        # Reshape for the layer
        weight = weight.view(weight_shape)
        bias = bias if self.bias_type == BiasType.WITH_BIAS.value else None
        # Wrap as nn.Parameter
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Activation
        x = LayerUtils.activation(x, self.activation_type, self.activation_param)

        # Layer normalization
        if self.norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (self.input_size,))

        # Dropout
        x = LayerUtils.dropout(x, self.dropout_rate, self.training)

        # Linear transformation
        x = F.linear(x, self.weight)

        # Normalize edge weights
        edge_index, edge_weight = gcn_norm(edge_index,
                                           add_self_loops=True,
                                           num_nodes=x.size(0), 
                                           dtype=x.dtype)
        # Edge weight check for jit scripting, which should never be None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1),
                                     dtype=x.dtype, 
                                     device='cuda')

        # Weighting by edge weights
        # Perform sparse matrix multiplication (A @ X)
        row, col = edge_index[0], edge_index[1] # Source nodes & Target nodes

        x_source_target = x[row] * edge_weight.unsqueeze(1)
        out = scatter(x_source_target, col, dim=0, dim_size=x.size(0), reduce='sum')

        # Add bias if applicable
        if self.bias_type == BiasType.WITH_BIAS.value:
            out = out + self.bias

        return out
    