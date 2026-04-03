from .layer import Layer, LayerUtils, LayerType
from .linear_layer import LinearLayer, LinearLayerUtils
from .conv_layer import ConvLayer, ConvLayerUtils
from .gcn_layer import GCNLayer, GCNLayerUtils
from .gat_layer import GATLayer, GATLayerUtils
from .embedding_layer import EmbeddingLayer, EmbeddingLayerUtils
from .multihead_atteention_layer import MultiheadAttentionLayer, MultiheadAttentionLayerUtils
from .kan_layer import KANLayer, KANLayerUtils
from typing import Dict, List, Tuple, Optional
from optimization import jit_script
import torch

# Layer class dictionary, access via "layer_type" value in the layer structure
LAYER_CLS_DICT: Dict[int, type[Layer]] = {
    LayerType.LINEAR.value: LinearLayer,
    LayerType.CONV.value: ConvLayer,
    LayerType.GCN.value: GCNLayer,
    LayerType.GAT.value: GATLayer,
    LayerType.EMBEDDING.value: EmbeddingLayer,
    LayerType.MHA.value: MultiheadAttentionLayer,
    LayerType.KAN.value: KANLayer
}

@jit_script
def layer_retrieve_shapes(layer_type: int, 
                          layer_structure: Dict[str, float]
                          ) -> Tuple[List[int], List[List[int]]]:
    '''
    Retrieve the lengths and shapes of the layer parameters.
    '''
    if layer_type == LayerType.LINEAR.value:
        return LinearLayerUtils.retrieve_shapes(layer_structure)
    elif layer_type == LayerType.CONV.value:
        return ConvLayerUtils.retrieve_shapes(layer_structure)
    elif layer_type == LayerType.GCN.value:
        return GCNLayerUtils.retrieve_shapes(layer_structure)
    elif layer_type == LayerType.GAT.value:
        return GATLayerUtils.retrieve_shapes(layer_structure)
    elif layer_type == LayerType.EMBEDDING.value:
        return EmbeddingLayerUtils.retrieve_shapes(layer_structure)
    elif layer_type == LayerType.MHA.value:
        return MultiheadAttentionLayerUtils.retrieve_shapes(layer_structure)
    elif layer_type == LayerType.KAN.value:
        return KANLayerUtils.retrieve_shapes(layer_structure)
    else:
        raise ValueError(f"Invalid layer type: {layer_type}.")

@jit_script
def layer_retrieve_required_arange_size(layer_type: int, 
                                        layer_structure: Dict[str, float]
                                        ) -> int:
    '''
    Retrieve the size of the arange tensor required for encoding the layer.
    '''
    if layer_type == LayerType.LINEAR.value:
        return LinearLayerUtils.retrieve_required_arange_size(layer_structure)
    elif layer_type == LayerType.CONV.value:
        return ConvLayerUtils.retrieve_required_arange_size(layer_structure)
    elif layer_type == LayerType.GCN.value:
        return GCNLayerUtils.retrieve_required_arange_size(layer_structure)
    elif layer_type == LayerType.GAT.value:
        return GATLayerUtils.retrieve_required_arange_size(layer_structure)
    elif layer_type == LayerType.EMBEDDING.value:
        return EmbeddingLayerUtils.retrieve_required_arange_size(layer_structure)
    elif layer_type == LayerType.MHA.value:
        return MultiheadAttentionLayerUtils.retrieve_required_arange_size(layer_structure)
    elif layer_type == LayerType.KAN.value:
        return KANLayerUtils.retrieve_required_arange_size(layer_structure)
    else:
        raise ValueError(f"Invalid layer type: {layer_type}.")

@jit_script
def layer_encode_index(layer_type: int,
                       layer_structure: Dict[str, float],
                       shared_element: torch.Tensor,
                       shared_memory: torch.Tensor,
                       unique_memory: Dict[str, List[torch.Tensor]],
                       arange_tensor: torch.Tensor
                       ) -> None:
    '''
    Encode the index of the layer to a preallocated memory, which should be pre-sliced.
    In this basic function, it only encodes the layer index and layer type.
    '''
    if layer_type == LayerType.LINEAR.value:
        LinearLayerUtils.encode_index(layer_structure,
                                      shared_element,
                                      shared_memory,
                                      unique_memory,
                                      arange_tensor)
    elif layer_type == LayerType.CONV.value:
        ConvLayerUtils.encode_index(layer_structure,
                                    shared_element,
                                    shared_memory,
                                    unique_memory,
                                    arange_tensor)
    elif layer_type == LayerType.GCN.value:
        GCNLayerUtils.encode_index(layer_structure,
                                   shared_element,
                                   shared_memory,
                                   unique_memory,
                                   arange_tensor)
    elif layer_type == LayerType.GAT.value:
        GATLayerUtils.encode_index(layer_structure,
                                   shared_element,
                                   shared_memory,
                                   unique_memory,
                                   arange_tensor)
    elif layer_type == LayerType.EMBEDDING.value:
        EmbeddingLayerUtils.encode_index(layer_structure,
                                          shared_element,
                                          shared_memory,
                                          unique_memory,
                                          arange_tensor)
    elif layer_type == LayerType.MHA.value:
        MultiheadAttentionLayerUtils.encode_index(layer_structure,
                                                  shared_element,
                                                  shared_memory,
                                                  unique_memory,
                                                  arange_tensor)
    elif layer_type == LayerType.KAN.value:
        KANLayerUtils.encode_index(layer_structure,
                                    shared_element,
                                    shared_memory,
                                    unique_memory,
                                    arange_tensor)
    else:
        raise ValueError(f"Invalid layer type: {layer_type}.")

@jit_script
def layer_get_params_initial_statistic(layer_type: int,
                                       layer_structure: Dict[str, float],
                                       ) -> List[Tuple[float, float]]:
    '''
    Get the initial mean and std of the layer parameters.
    '''
    if layer_type == LayerType.LINEAR.value:
        return LinearLayerUtils.get_params_initial_statistic(layer_structure)
    elif layer_type == LayerType.CONV.value:
        return ConvLayerUtils.get_params_initial_statistic(layer_structure)
    elif layer_type == LayerType.GCN.value:
        return GCNLayerUtils.get_params_initial_statistic(layer_structure)
    elif layer_type == LayerType.GAT.value:
        return GATLayerUtils.get_params_initial_statistic(layer_structure)
    elif layer_type == LayerType.EMBEDDING.value:
        return EmbeddingLayerUtils.get_params_initial_statistic(layer_structure)
    elif layer_type == LayerType.MHA.value:
        return MultiheadAttentionLayerUtils.get_params_initial_statistic(layer_structure)
    elif layer_type == LayerType.KAN.value:
        return KANLayerUtils.get_params_initial_statistic(layer_structure)
    else:
        raise ValueError(f"Invalid layer type: {layer_type}.")

@jit_script
def layer_apply_params_1i(layer_type: int,
                          x: torch.Tensor,
                          layer_structure: Dict[str, float],
                          layer_params: List[torch.Tensor],
                          layer_param_shapes: List[List[int]],
                          training: bool
                          ) -> torch.Tensor:                          
    '''
    Apply the layer to the 1 input tensor.
    '''
    if layer_type == LayerType.LINEAR.value:
        return LinearLayerUtils.apply_params(
            x=x,
            layer_structure=layer_structure,
            layer_params=layer_params,
            layer_param_shapes=layer_param_shapes,
            training=training
        )
    elif layer_type == LayerType.CONV.value:
        return ConvLayerUtils.apply_params(
            x=x,
            layer_structure=layer_structure,
            layer_params=layer_params,
            layer_param_shapes=layer_param_shapes,
            training=training
        )
    elif layer_type == LayerType.EMBEDDING.value:
        return EmbeddingLayerUtils.apply_params(
            x=x,
            layer_structure=layer_structure,
            layer_params=layer_params,
            layer_param_shapes=layer_param_shapes,
            training=training
        )
    elif layer_type == LayerType.KAN.value:
        return KANLayerUtils.apply_params(
            x=x,
            layer_structure=layer_structure,
            layer_params=layer_params,
            layer_param_shapes=layer_param_shapes,
            training=training
        )
    else:
        raise ValueError(f"Invalid layer type: {layer_type}.")
    
@jit_script
def layer_apply_params_2i(layer_type: int,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          layer_structure: Dict[str, float],
                          layer_params: List[torch.Tensor],
                          layer_param_shapes: List[List[int]],
                          training: bool
                          ) -> torch.Tensor:                          
    '''
    Apply the layer to the 2 input tensors.
    '''
    if layer_type == LayerType.GCN.value:
        return GCNLayerUtils.apply_params(
            x=x,
            edge_index=y,
            layer_structure=layer_structure,
            layer_params=layer_params,
            layer_param_shapes=layer_param_shapes,
            training=training
        )
    elif layer_type == LayerType.GAT.value:
        return GATLayerUtils.apply_params(
            x=x,
            edge_index=y,
            layer_structure=layer_structure,
            layer_params=layer_params,
            layer_param_shapes=layer_param_shapes,
            training=training
        )
    else:
        raise ValueError(f"Invalid layer type: {layer_type}.")
