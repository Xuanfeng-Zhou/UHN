'''
Embedding LayerUtils encoding & forwarding and modularization
'''

from .layer import *


class EmbeddingLayerUtils:
    '''
    Embedding layer encoding & forwarding and modularization
    '''
    @staticmethod
    @jit_script
    def get_layer_mode(layer_idx: int,
                         embedding_num: int,
                         embedding_dim: int,
                         max_sequence_length: int,
                         dropout_rate: float,
                         initialization_type: int
                         ) -> Dict[str, float]:
        '''
        Get a layer mode dictionary for embedding layer.
        '''
        layer_mode: Dict[str, float] = LayerUtils.get_layer_mode()
        layer_mode.update({
            "layer_idx": float(layer_idx),
            "layer_type": float(LayerType.EMBEDDING.value),
            "output_size": float(embedding_dim),
            "dropout_rate": dropout_rate,
            "embedding_num": float(embedding_num),
            "max_sequence_length": float(max_sequence_length),
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
        For embedding layer, encode param type, output idx and embedding idx 
            besides the basic index encoding in the LayerUtils class.
        '''
        LayerUtils.encode_index(layer_structure, shared_element, 
                             shared_memory, unique_memory, arange_tensor)

        # Retrieve the layer structure
        embedding_num: int = int(layer_structure["embedding_num"])
        embedding_dim: int = int(layer_structure["output_size"])
        max_sequence_length: int = int(layer_structure["max_sequence_length"])
        
        # Retrieve pre-allocated memory for encoding
        param_type_embed, param_type_pos = unique_memory[IndexType.Unique.PARAM_TYPE.name]
        output_idx_embed, output_idx_pos = unique_memory[IndexType.Unique.OUTPUT_IDX.name]
        embedding_idx_embed, _ = unique_memory[IndexType.Unique.EMBEDDING_IDX.name]
        _, sequence_idx_pos = unique_memory[IndexType.Unique.SEQUENCE_IDX.name]

        # Assign param type encoding
        param_type_embed.fill_(ParamType.Embedding.EMBEDDING_WEIGHTS.value)
        param_type_pos.fill_(ParamType.Embedding.POSITION_WEIGHTS.value)
        # Reshape the weights encoding for the layer
        output_idx_embed = output_idx_embed.view(embedding_num, embedding_dim)
        embedding_idx_embed = embedding_idx_embed.view(embedding_num, embedding_dim)
        output_idx_pos = output_idx_pos.view(1, max_sequence_length, embedding_dim)
        sequence_idx_pos = sequence_idx_pos.view(1, max_sequence_length, embedding_dim)
        # Retrieve values by slicing arange tensor
        slice_output_idx = arange_tensor[:embedding_dim]
        slice_embedding_idx = arange_tensor[:embedding_num]
        slice_sequence_idx = arange_tensor[:max_sequence_length]
        # Copy the values to the pre-allocated memory
        # Embedding weights
        output_idx_embed.copy_(slice_output_idx.view(1, -1))
        embedding_idx_embed.copy_(slice_embedding_idx.view(-1, 1))
        # Positional encoding weights
        output_idx_pos.copy_(slice_output_idx.view(1, 1, -1))
        sequence_idx_pos.copy_(slice_sequence_idx.view(1, -1, 1))

    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding this embedding layer.
        '''
        embedding_num: int = int(layer_structure["embedding_num"])
        embedding_dim: int = int(layer_structure["output_size"])
        max_sequence_length: int = int(layer_structure["max_sequence_length"])
        return max(embedding_num, embedding_dim, max_sequence_length,
                   LayerUtils.retrieve_required_arange_size(layer_structure))
    
    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of embedding layer parameters.
        '''
        layer_lens, layer_shapes = LayerUtils.retrieve_shapes(layer_structure)

        # Retrieve the layer structure
        embedding_num: int = int(layer_structure["embedding_num"])
        embedding_dim: int = int(layer_structure["output_size"])
        max_sequence_length: int = int(layer_structure["max_sequence_length"])
        # Param of embedding weights
        layer_lens.append(embedding_num * embedding_dim)
        layer_shapes.append([embedding_num, embedding_dim])
        # Param of positional encoding weights
        layer_lens.append(max_sequence_length * embedding_dim)
        layer_shapes.append([1, max_sequence_length, embedding_dim])

        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of this embedding layer parameters.
        Use Normal(0, 1) for weights initialization.
        '''
        layer_stats = LayerUtils.get_params_initial_statistic(layer_structure)

        # Retrieve the layer structure
        init_type: int = int(layer_structure["initialization_type"])

        if init_type == InitializationType.DEFAULT.value:
            # Embedding layer weights are initialized with Normal(0, 1)
            layer_stats.append((0.0, 1.0))
            # Positional encoding weights are initialized as zeros
            layer_stats.append((0.0, 0.0))
        elif init_type == InitializationType.ZERO.value:
            # Embedding layer weights are initialized as zeros
            layer_stats.append((0.0, 0.0))
            # Positional encoding weights are initialized as zeros
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
        Apply the embedding layer to the input tensor.
        '''
        x = LayerUtils.apply_params(x, layer_structure, layer_params,
                                 layer_param_shapes, training)
        
        # Retrieve the layer structure
        dropout_rate: float = layer_structure["dropout_rate"]

        # Retrieve the weights
        embed_weight, pos_weight = layer_params
        embed_weight_shape, pos_weight_shape = layer_param_shapes   
        # Reshape the weights
        embed_weight = embed_weight.view(embed_weight_shape)
        pos_weight = pos_weight.view(pos_weight_shape)

        # Apply the embedding
        x = F.embedding(x, embed_weight)

        # Add positional encoding
        x_pos = pos_weight[:, :x.shape[1], :]
        x = x + x_pos

        # Apply dropout to the sum of input embedding and positional encoding
        x = LayerUtils.dropout(x, dropout_rate, training)

        return x

class EmbeddingLayer(Layer):
    '''
    Embedding layer encoding & forwarding and modularization
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the linear layer.
        '''    
        # Call the apply_params function as forward
        return EmbeddingLayerUtils.apply_params(x=x, 
                                             layer_structure=self.layer_structure, 
                                             layer_params=list(self.layer_params), 
                                             layer_param_shapes=self.layer_param_shapes, 
                                             training=self.training)

class EmbeddingLayerWrapper(nn.Module):
    '''
    A wrapper class for EmbeddingLayer for the convenience of ONNX export.
    '''
    def __init__(self,
                 embedding_layer: EmbeddingLayer
                 ) -> None:
        super(EmbeddingLayerWrapper, self).__init__()
        layer_structure = embedding_layer.layer_structure

        # Retrieve the layer structure
        self.dropout_rate: float = layer_structure["dropout_rate"]

        # Retrieve the weights
        embed_weight, pos_weight = embedding_layer.layer_params
        embed_weight_shape, pos_weight_shape = embedding_layer.layer_param_shapes
        # Reshape the weights
        embed_weight = embed_weight.view(embed_weight_shape)
        pos_weight = pos_weight.view(pos_weight_shape)
        # Wrap as nn.Parameter
        self.embed_weight = nn.Parameter(embed_weight)
        self.pos_weight = nn.Parameter(pos_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the embedding
        x = F.embedding(x, self.embed_weight)

        # Add positional encoding
        x_pos = self.pos_weight[:, :x.shape[1], :]
        x = x + x_pos

        # Apply dropout to the sum of input embedding and positional encoding
        x = LayerUtils.dropout(x, self.dropout_rate, self.training)

        return x
