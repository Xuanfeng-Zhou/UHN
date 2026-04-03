'''
Layer encoding & forwarding and modularization

Structure of Layers in the model

0. Shared 
Structural Parameters:
    - Layer idx
    - Layer type
Index:
    - Layer idx
    - Layer type
    - Param type
    - Output idx / Output channel idx / Output Embedding channel idx

1. Linear layer
Params (2):
    - Weights
    - Biases
Structural Parameters:
    - Bias type: With Bias / None 
    - Normalization type: Activation Norm
    - Shortcut type: straight shortcut / none 
    - Output size
    - Input size
    - Activation type
    - Activation parameter
    - Dropout rate
    - Input pooling or reshape type
Index:
    - Input idx

2. Convolutional layer
Params (2):
    - Weights
    - Biases
Structural Parameters:
    - Bias type: With Bias / None 
    - Normalization type: Activation Norm
    - Number of groups for group norm
    - Shortcut type: straight shortcut / none
    - Output channel dimension
    - Input channel dimension
    - Kernel size
    - Activation type
    - Activation parameter
    - Dropout rate
    - Stage-wise pooling type
Index:
    - Input channel idx
    - Kernel H idx
    - Kernel W idx
    
3. GCN Convolutional layer
Params (2):
    - Weights
    - Biases
Structural Parameters:
    - Bias type: With Bias / None 
    - Normalization type: Activation Norm
    - Output channel dimension
    - Input channel dimension
    - Activation type
    - Activation parameter
    - Dropout rate
Index:
    - Input channel idx

4. GAT Convolutional layer
Params (4):
    - Weights
    - Biases
    - Attention weights (src)
    - Attention weights (dst)
Structural Parameters:
    - Bias type: With Bias / None 
    - Normalization type: Activation Norm
    - Output channel dimension
    - Input channel dimension
    - Activation type
    - Activation parameter
    - Dropout rate
    - Number of heads
    - Head concat type: Concat / Avg
Index:
    - Input channel idx

5. Embedding layer
Params (2):
    - Embedding Weights
    - Position Weights
Structural Parameters:
    - Embedding number
    - Embedding dimension
    - Dropout rate
    - Max sequence length
Index:
    - Embedding idx
    - Sequence idx

6. Multi-head Attention layer
Params (8):
    - Projection weights (Q, K, V)
    - Projection biases (Q, K, V)
    - Out Projection weights
    - Out Projection biases
Structural Parameters:
    - Bias type: With Bias / None 
    - Normalization type: Activation Norm
    - Shortcut type: straight shortcut / none 
    - Output dimension (Embedding dimension)
    - Input dimension (also Embedding dimension)
    - Activation type
    - Activation parameter
    - Dropout rate
    - Number of heads
Index:
    - Input Embedding channel idx
    
7. KAN Linear layer
Params (7):
    - Base weights
    - Base biases
    - Spline weights
    - Spline scales
    - Grid lower bounds
    - Grid lengths
    - Grid knots
Structural Parameters:
    - Bias type: With Bias / None
    - Output size
    - Input size
    - Base activation type
    - Base activation parameter
    - Grid size
    - Spline order
Index:
    - Input idx
    - Grid idx

*. Miscellaneous
Structural Parameters (not necessarily structural):
    - Initialization type
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math
from optimization import jit_script

# Layer Type
class LayerType(Enum):
    '''
    Layer Type Enum
    '''
    LINEAR = 0
    CONV = 1
    GCN = 2
    GAT = 3
    EMBEDDING = 4
    MHA = 5
    KAN = 6

# Bias Type
class BiasType(Enum):
    '''
    Bias Type Enum
    '''
    NONE = 0
    WITH_BIAS = 1

# Norm Type
class NormType(Enum):
    '''
    Norm Type Enum
    '''
    NONE = 0
    ACTIVATION_NORM = 1

# Shortcut Type
class ShortcutType(Enum):
    '''
    Shortcut Type Enum
    '''
    NONE = 0
    STRAIGHT = 1

# Activation Type
class ActivationType(Enum):
    '''
    Activation Type Enum
    '''
    NONE = 0
    LEAKY_RELU = 1
    ELU = 2
    SILU = 3

# CNN Input Pooling or Reshape Type
class InputPoolingReshapeType(Enum):
    '''
    CNN Input Pooling or Reshape Type Enum
    '''
    NONE = 0
    # Avg pooling for 4d input then reshape to (batch_size, dim)
    AVG_POOLING = 1
    # Max pooling for 4d input then reshape to (batch_size, dim)
    MAX_POOLING = 2
    # Reshape to (batch_size, dim)
    RESHAPE_TO_2D = 3
    # Avg along the sequence length (dim 1)
    AVG_SEQUENCE = 4
    # Use the first token's representation (dim 1)
    FIRST_TOKEN = 5

# CNN Stage-wise Pooling Type
class StageWisePoolingType(Enum):
    '''
    CNN Stage-wise Pooling Type Enum
    '''
    NONE = 0
    AVG_POOLING = 1
    MAX_POOLING = 2
    CONV_POOLING = 3

# Head concat type
class HeadConcatType(Enum):
    '''
    Head Concat Type Enum
    '''
    CONCAT = 0
    AVG = 1

# Initialization Type
class InitializationType(Enum):
    '''
    Initialization Type Enum
    '''
    # PyTorch default
    DEFAULT = 0
    # Zero
    ZERO = 1

# Param Type
class ParamType:
    class Linear(Enum):
        '''
        Linear Layer Param Type
        '''
        WEIGHTS = 0
        BIASES = 1

    class Conv(Enum):
        '''
        Convolutional Layer Param Type
        '''
        WEIGHTS = 0
        BIASES = 1
    
    class GCN(Enum):
        '''
        GCN Convolutional Layer Param Type
        '''
        WEIGHTS = 0
        BIASES = 1
    
    class GAT(Enum):
        '''
        GAT Convolutional Layer Param Type
        '''
        WEIGHTS = 0
        BIASES = 1
        ATTENTION_WEIGHTS_SRC = 2
        ATTENTION_WEIGHTS_DST = 3

    class Embedding(Enum):
        '''
        Embedding Layer Param Type
        '''
        EMBEDDING_WEIGHTS = 0
        POSITION_WEIGHTS = 1

    class MHA(Enum):
        '''
        Multi-head Attention Layer Param Type
        '''
        PROJECTION_WEIGHTS_Q = 0
        PROJECTION_WEIGHTS_K = 1
        PROJECTION_WEIGHTS_V = 2
        PROJECTION_BIASES_Q = 3
        PROJECTION_BIASES_K = 4
        PROJECTION_BIASES_V = 5
        OUT_PROJECTION_WEIGHTS = 6
        OUT_PROJECTION_BIASES = 7
    
    class KAN(Enum):
        '''
        KAN Linear Layer Param Type
        '''
        BASE_WEIGHTS = 0
        BASE_BIASES = 1
        SPLINE_WEIGHTS = 2
        SPLINE_SCALES = 3
        GRID_LOWER_BOUNDS = 4
        GRID_LENGTHS = 5
        GRID_KNOTS = 6

# Type of index encoding
class IndexType:
    '''
    Index Type Enum
    '''
    class Shared(Enum):
        '''
        Shared Index Type Enum
        '''
        LAYER_IDX = 0
        LAYER_TYPE = 1

    class Unique(Enum):
        '''
        Unique Index Type Enum
        '''
        PARAM_TYPE = 2
        OUTPUT_IDX = 3
        INPUT_IDX = 4
        KERNEL_H_IDX = 5
        KERNEL_W_IDX = 6
        EMBEDDING_IDX = 7
        SEQUENCE_IDX = 8
        GRID_IDX = 9
    
    @staticmethod
    def get_shared_length() -> int:
        '''
        Get the length of the shared index. Hardcoded for the benifit of jit.script.
        '''
        return 2
    
    @staticmethod
    def get_unique_length() -> int:
        '''
        Get the length of the unique index. Hardcoded for the benifit of jit.script.
        '''
        return 8
    
    @staticmethod
    def get_length() -> int:
        '''
        Get the length of the index. Hardcoded for the benifit of jit.script.
        '''
        return 10
    
    @staticmethod
    def get_unique_dict() -> Dict[str, int]:
        '''
        Get the unique index dictionary. Hardcoded for the benifit of jit.script.
        '''
        return {
            "PARAM_TYPE": 2,
            "OUTPUT_IDX": 3,
            "INPUT_IDX": 4,
            "KERNEL_H_IDX": 5,
            "KERNEL_W_IDX": 6,
            "EMBEDDING_IDX": 7,
            "SEQUENCE_IDX": 8,
            "GRID_IDX": 9
        }

class LayerUtils:
    '''
    Layer utils class for encoding and forwarding
    '''
    @staticmethod
    @jit_script
    def get_layer_mode() -> Dict[str, float]:
        '''
        Get a layer mode dictionary for the layer.
        '''
        layer_mode: Dict[str, float] = {}
        return layer_mode

    @staticmethod
    @jit_script
    def encode_layer_structure(structures_update: Optional[Dict[str, float]] = None) -> List[float]:
        '''
        Encode the structure of the layer to a list of float values.
        The default value is set to 0.0 for all the parameters.
        '''
        structures: Dict[str, float] = {
            # --- Shared Parameters ---
            "layer_idx": 0.0,
            "layer_type": 0.0,

            # --- Linear Layer Parameters ---
            # Bias type: With Bias / None
            "bias_type": 0.0,
            # Normalization type: Activation Norm / None
            "norm_type": 0.0,
            # Shortcut type: straight shortcut / none
            "shortcut_type": 0.0,
            # Output size / Output channel dimension / Embedding dimension
            "output_size": 0.0,
            # Input size / Input channel dimension
            "input_size": 0.0,
            # Activation type (also for base activation type of KAN)
            "activation_type": 0.0,
            # Activation parameter (e.g. leaky rate, also for base activation parameter of KAN)
            "activation_param": 0.0,
            # Dropout rate
            "dropout_rate": 0.0,
            # Input pooling or reshape type
            "input_pooling_reshape_type": 0.0,

            # --- Convolutional Layer Parameters ---
            # Number of groups for group norm
            "group_num": 0.0,
            # Kernel size
            "kernel_size": 0.0,
            # Stage-wise pooling type
            "stage_wise_pooling_type": 0.0,

            # --- GCN Convolutional Layer Parameters ---
            # None

            # --- GAT Convolutional Layer Parameters ---
            # Number of heads (For GAT and Multi-head Attention)
            "num_heads": 0.0,
            # Head concat type: Concat / Avg
            "head_concat_type": 0.0,

            # --- Embedding Layer Parameters ---
            # Embedding Number
            "embedding_num": 0.0,
            # Max sequence length
            "max_sequence_length": 0.0,

            # --- Multi-head Attention Layer Parameters ---
            # None

            # --- KAN Linear Layer Parameters ---
            # Grid size
            "grid_size": 0.0,
            # Spline order
            "spline_order": 0.0,

            # --- Miscellaneous Parameters ---
            # Initialization type
            "initialization_type": 0.0,
        }

        # Check if the updating values are legitimate and update the structure
        if structures_update is not None:
            for updated_key in structures_update:
                if updated_key not in structures:
                    raise ValueError(f"Invalid attribute in layer structure: {updated_key}")
            # Update the structure with the given values
            structures.update(structures_update)
        return list(structures.values())
    
    @staticmethod
    @jit_script
    def encode_index(layer_structure: Dict[str, float],
                     shared_element: torch.Tensor,
                     shared_memory: torch.Tensor,
                     unique_memory: Dict[str, List[torch.Tensor]],
                     arange_tensor: torch.Tensor
                     ) -> None:
        '''
        Encode the index of the layer to a preallocated memory, which should be pre-sliced.
        In this basic function, it only encodes the layer index and layer type.
        
        layer_structure: the structure of the layer.
        shared_element: shared element to be encoded for the layer.
        shared_memory: a sliced preallocated memories for encoding the shared element.
        unique_memory: a dictionary of sliced preallocated memories for encoding the unique elements.
        arange_tensor: a preallocated memory with arange tensor.
        
        Index (Should be initialized as -1):
            - Layer idx
            - Layer type
            - Param type
            - Output idx / Output channel idx / Output Embedding channel idx
            - Input idx / Input channel idx / Input Embedding channel idx
            - Kernel H idx
            - Kernel W idx
            - Embedding idx
            - Sequence idx
            - Grid idx
        '''
        # Expand the shared memory into shape (batch, #elements) and assign
        shared_memory.copy_(shared_element.view(1, -1))
    
    @staticmethod
    @jit_script
    def retrieve_required_arange_size(layer_structure: Dict[str, float]) -> int:
        '''
        Retrieve the size of the arange tensor required for encoding the layer.

        layer_structure: the structure of the layer.
        '''
        return 0

    @staticmethod
    @jit_script
    def retrieve_shapes(layer_structure: Dict[str, float]
                        ) -> Tuple[List[int], List[List[int]]]:
        '''
        Retrieve the lengths and shapes of the layer parameters.
        '''
        layer_lens: List[int] = []
        layer_shapes: List[List[int]] = []
        return layer_lens, layer_shapes

    @staticmethod
    @jit_script
    def get_params_initial_statistic(layer_structure: Dict[str, float],
                                    ) -> List[Tuple[float, float]]:
        '''
        Get the initial mean and std of the layer parameters.
        '''
        layer_stats: List[Tuple[float, float]] = []
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
        Apply the layer to the input tensor.
        Straight through in this basic function.
        '''
        # Apply straight through for the layer
        return x

    @staticmethod
    @jit_script
    def activation(x: torch.Tensor,
                   activation_type: int,
                   activation_param: float,
                   ) -> torch.Tensor:
        '''
        Activation function for the layer.
        '''
        if activation_type == ActivationType.LEAKY_RELU.value:
            return F.leaky_relu(x, negative_slope=activation_param)
        elif activation_type == ActivationType.ELU.value:
            return F.elu(x, alpha=activation_param)
        elif activation_type == ActivationType.SILU.value:
            return F.silu(x)
        else:
            return x
        
    @staticmethod
    @jit_script
    def dropout(x: torch.Tensor,
                dropout_rate: float,
                training: bool,
                ) -> torch.Tensor:
        '''
        Dropout function for the layer.
        '''
        if dropout_rate > 0:
            return F.dropout(x, p=dropout_rate, training=training)
        else:
            return x

class Layer(nn.Module):
    '''
    Layer class for encoding and forwarding
    '''
    @classmethod
    def modularize(cls, 
                   layer_structure: Dict[str, float],
                   layer_params: List[torch.Tensor],
                   layer_param_shapes: List[List[int]]
                   ) -> nn.Module:
        '''
        Modularize the layer.
        '''
        return cls(layer_structure, layer_params, layer_param_shapes)

    def __init__(self,
                 layer_structure: Dict[str, float],
                 layer_params: List[torch.Tensor],
                 layer_param_shapes: List[List[int]],
                 ) -> None:
        '''
        Initialize the layer as a module.
        '''
        super(Layer, self).__init__()

        # Save the layer parameters
        self.layer_params = nn.ParameterList([nn.Parameter(param) for param in layer_params])
        self.layer_structure = layer_structure
        self.layer_param_shapes = layer_param_shapes

    def state_dict(self, **kwargs):
        '''
        Save the layer param shapes additionally.
        '''
        state = super(Layer, self).state_dict(**kwargs)
        state['layer_structure'] = self.layer_structure
        state['layer_param_shapes'] = self.layer_param_shapes
        return state
    
    def load_state_dict(self, state_dict, **kwargs):
        '''
        Load the layer param shapes additionally.
        '''
        self.layer_structure = state_dict.pop('layer_structure')
        self.layer_param_shapes = state_dict.pop('layer_param_shapes')
        super(Layer, self).load_state_dict(state_dict, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward the layer.
        '''
        # Call the apply_params function as forward
        return LayerUtils.apply_params(x=x, 
                                       layer_structure=self.layer_structure, 
                                       layer_params=list(self.layer_params), 
                                       layer_param_shapes=self.layer_param_shapes, 
                                       training=self.training)
