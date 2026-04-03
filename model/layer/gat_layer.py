'''
Graph Attention Layer
'''

from .layer import *
from torch_geometric.utils import add_self_loops, softmax, scatter

class GATLayerUtils:
    '''
    Graph Attention layer encoding & forwarding and modularization
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
                       num_heads: int,
                       head_concat_type: int,
                       initialization_type: int
                       ) -> Dict[str, float]:
        '''
        Get a layer mode dictionary for graph attention layer.
        '''
        layer_mode: Dict[str, float] = LayerUtils.get_layer_mode()
        layer_mode.update({
            "layer_idx": float(layer_idx),
            "layer_type": float(LayerType.GAT.value),
            "bias_type": float(bias_type),
            "norm_type": float(norm_type),
            "output_size": float(output_dim),
            "input_size": float(input_dim),
            "activation_type": float(activation_type),
            "activation_param": activation_param,
            "dropout_rate": dropout_rate,
            "num_heads": float(num_heads),
            "head_concat_type": float(head_concat_type),
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
        For GAT layer, encode param type, output idx and input idx 
            besides the basic index encoding in the LayerUtils class.
        '''
        LayerUtils.encode_index(layer_structure, shared_element, 
                             shared_memory, unique_memory, arange_tensor)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])
        num_heads: int = int(layer_structure["num_heads"])
        head_concat_type: int = int(layer_structure["head_concat_type"])

        # Retrieve pre-allocated memory for encoding
        param_type_weight, param_type_bias, param_type_att_src, param_type_att_dst = unique_memory[IndexType.Unique.PARAM_TYPE.name]
        output_idx_weight, output_idx_bias, output_idx_att_src, output_idx_att_dst = unique_memory[IndexType.Unique.OUTPUT_IDX.name]
        input_idx_weight, _, _, _ = unique_memory[IndexType.Unique.INPUT_IDX.name]

        # Assign param type encoding
        param_type_weight.fill_(ParamType.GAT.WEIGHTS.value)
        # Reshape the weights encoding for the layer
        output_idx_weight = output_idx_weight.view(num_heads * output_dim, input_dim)
        input_idx_weight = input_idx_weight.view(num_heads * output_dim, input_dim)
        # Retrieve values by slicing arange tensor
        slice_output_idx = arange_tensor[:num_heads * output_dim]
        slice_input_idx = arange_tensor[:input_dim]
        # Copy the values to the pre-allocated memory
        output_idx_weight.copy_(slice_output_idx.view(-1, 1))
        input_idx_weight.copy_(slice_input_idx.view(1, -1))

        # Encoding bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            # Assign param type encoding
            param_type_bias.fill_(ParamType.GAT.BIASES.value)
            # Encoding output idx and ignore input idx
            if head_concat_type == HeadConcatType.CONCAT.value:
                # Output bias with shape (num_heads * output_dim,)
                output_idx_bias.copy_(slice_output_idx)
            else:
                # Output bias with shape (output_dim,)
                output_idx_bias.copy_(arange_tensor[:output_dim])

        # Encoding attention weights for source and target
        # Assign param type encoding
        param_type_att_src.fill_(ParamType.GAT.ATTENTION_WEIGHTS_SRC.value)
        param_type_att_dst.fill_(ParamType.GAT.ATTENTION_WEIGHTS_DST.value)
        # Encoding output idx and ignore input idx
        output_idx_att_src.copy_(slice_output_idx)
        output_idx_att_dst.copy_(slice_output_idx)

    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding this gat layer.
        '''
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])
        num_heads: int = int(layer_structure["num_heads"])
        return max(output_dim, input_dim, num_heads * output_dim,
                   LayerUtils.retrieve_required_arange_size(layer_structure))
    
    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of GAT layer parameters.
        '''
        layer_lens, layer_shapes = LayerUtils.retrieve_shapes(layer_structure)

        # Retrieve the layer structure
        bias_type: int = int(layer_structure["bias_type"])
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])
        num_heads: int = int(layer_structure["num_heads"])
        head_concat_type: int = int(layer_structure["head_concat_type"])

        # Param of weight
        layer_lens.append(num_heads * output_dim * input_dim)
        layer_shapes.append([num_heads * output_dim, input_dim])
        # Param of bias
        if bias_type == BiasType.WITH_BIAS.value:
            if head_concat_type == HeadConcatType.CONCAT.value:
                bias_num = num_heads * output_dim
            else:
                bias_num = output_dim
        else:
            bias_num = 0
        layer_lens.append(bias_num)
        layer_shapes.append([bias_num])
        # Param of attention weights for source
        layer_lens.append(num_heads * output_dim)
        layer_shapes.append([1, num_heads, output_dim])
        # Param of attention weights for target
        layer_lens.append(num_heads * output_dim)
        layer_shapes.append([1, num_heads, output_dim])

        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of this gat layer parameters.
        Use Glort initialization for weights, attentions and zero initialization for biases.
        '''
        layer_stats = LayerUtils.get_params_initial_statistic(layer_structure)

        # Retrieve the layer structure
        output_dim: int = int(layer_structure["output_size"])
        input_dim: int = int(layer_structure["input_size"])
        num_heads: int = int(layer_structure["num_heads"])
        init_type: int = int(layer_structure["initialization_type"])

        if init_type == InitializationType.DEFAULT.value:
            # Calculate the target mean and std for Glort initialization of weights
            target_weight_mean = 0.0
            target_weight_std = math.sqrt(2.0 / (input_dim + output_dim * num_heads))
            layer_stats.append((target_weight_mean, target_weight_std))
            # Zero initialization for biases
            layer_stats.append((0.0, 0.0))
            # Glort initialization for attention weights
            target_att_mean = 0.0
            target_att_std = math.sqrt(2.0 / (output_dim + num_heads))
            # Attention weights for source
            layer_stats.append((target_att_mean, target_att_std))
            # Attention weights for target
            layer_stats.append((target_att_mean, target_att_std))
        elif init_type == InitializationType.ZERO.value:
            # Zero initialization for weights
            layer_stats.append((0.0, 0.0))
            # Zero initialization for biases
            layer_stats.append((0.0, 0.0))
            # Zero initialization for attention weights
            layer_stats.append((0.0, 0.0))
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
        Apply the GAT layer to the input tensor.
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
        num_heads: int = int(layer_structure["num_heads"])
        head_concat_type: int = int(layer_structure["head_concat_type"])

        # Retrieve layer parameters
        weight, bias, att_src, att_dst = layer_params
        weight_shape, _, att_src_shape, att_dst_shape = layer_param_shapes
        # Reshape for the layer
        weight = weight.view(weight_shape)
        att_src = att_src.view(att_src_shape)
        att_dst = att_dst.view(att_dst_shape)

        # Activation
        x = LayerUtils.activation(x, activation_type, activation_param)

        # Layer normalization
        if norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (x.shape[-1],))

        # Dropout
        x = LayerUtils.dropout(x, dropout_rate, training)

        # Linear transformation and reshape the input for attention
        output_dim = att_src_shape[-1]
        x = F.linear(x, weight).view(-1, num_heads, output_dim)

        # Attention mechanism for each node (num_nodes, num_heads)
        alpha_src = (x * att_src).sum(dim=-1)
        alpha_dst = (x * att_dst).sum(dim=-1)
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index[0], edge_index[1] # Source nodes & Target nodes
        # Calculate attention scores for each edge (num_edges, num_heads)
        alpha = alpha_src[row] + alpha_dst[col]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # Attention normalization
        alpha = softmax(alpha, col, num_nodes=x.size(0))
        # Dropout attention scores
        alpha = LayerUtils.dropout(alpha, dropout_rate, training)

        # Aggregate neighbor features using attention coefficients
        x_source_target = x[row] * alpha.unsqueeze(2)
        out = scatter(x_source_target, col, dim=0, dim_size=x.size(0), reduce='sum')
        
        if head_concat_type == HeadConcatType.CONCAT.value:
            # Concatenate multi-heads
            out = out.view(-1, num_heads * output_dim)
        else:
            # Average multi-heads
            out = out.mean(dim=1)

        # Apply bias if applicable
        if bias_type == BiasType.WITH_BIAS.value:
            out = out + bias

        return out

class GATLayer(Layer):
    '''
    Graph Attention layer encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        '''
        Forward the input tensor through the GAT layer.
        '''
        # Call the apply_params function as forward
        return GATLayerUtils.apply_params(x=x, 
                                          edge_index=edge_index,
                                          layer_structure=self.layer_structure, 
                                          layer_params=list(self.layer_params), 
                                          layer_param_shapes=self.layer_param_shapes, 
                                          training=self.training)

class GATLayerWrapper(nn.Module):
    def __init__(self, gat_layer: GATLayer) -> None:
        super(GATLayerWrapper, self).__init__()
        layer_structure = gat_layer.layer_structure

        # Retrieve the layer structure
        self.bias_type: int = int(layer_structure["bias_type"])
        self.norm_type: int = int(layer_structure["norm_type"])
        self.activation_type: int = int(layer_structure["activation_type"])
        self.activation_param: float = layer_structure["activation_param"]
        self.dropout_rate: float = layer_structure["dropout_rate"]
        self.num_heads: int = int(layer_structure["num_heads"])
        self.head_concat_type: int = int(layer_structure["head_concat_type"])
        self.input_size: int = int(layer_structure["input_size"])

        # Retrieve layer parameters
        weight, bias, att_src, att_dst = gat_layer.layer_params
        weight_shape, _, att_src_shape, att_dst_shape = gat_layer.layer_param_shapes
        # Reshape for the layer
        weight = weight.view(weight_shape)
        bias = bias if self.bias_type == BiasType.WITH_BIAS.value else None
        att_src = att_src.view(att_src_shape)
        att_dst = att_dst.view(att_dst_shape)
        # Wrap as nn.Parameter
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
        self.att_src = nn.Parameter(att_src)
        self.att_dst = nn.Parameter(att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Activation
        x = LayerUtils.activation(x, self.activation_type, self.activation_param)

        # Layer normalization
        if self.norm_type == NormType.ACTIVATION_NORM.value:
            x = F.layer_norm(x, (self.input_size,))

        # Dropout
        x = LayerUtils.dropout(x, self.dropout_rate, self.training)

        # Linear transformation and reshape the input for attention
        output_dim = self.att_src.shape[-1]
        x = F.linear(x, self.weight).view(-1, self.num_heads, output_dim)

        # Attention mechanism for each node (num_nodes, num_heads)
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index[0], edge_index[1] # Source nodes & Target nodes
        # Calculate attention scores for each edge (num_edges, num_heads)
        alpha = alpha_src[row] + alpha_dst[col]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # FIXME: Seems to have a onnx conversion issue here, which leads to the exported model inexecutable
        # Attention normalization
        alpha = softmax(alpha, col, num_nodes=x.size(0))
        # Dropout attention scores
        alpha = LayerUtils.dropout(alpha, self.dropout_rate, self.training)

        # Aggregate neighbor features using attention coefficients
        x_source_target = x[row] * alpha.unsqueeze(2)
        out = scatter(x_source_target, col, dim=0, dim_size=x.size(0), reduce='sum')

        if self.head_concat_type == HeadConcatType.CONCAT.value:
            # Concatenate multi-heads
            out = out.view(-1, self.num_heads * output_dim)
        else:
            # Average multi-heads
            out = out.mean(dim=1)

        # Apply bias if applicable
        if self.bias_type == BiasType.WITH_BIAS.value:
            out = out + self.bias

        return out        
