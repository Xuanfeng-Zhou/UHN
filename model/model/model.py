'''
Model encoding & forward and modularization

Structure of Model Encoding

0. Shared
Global Parameters:
    - Model type
    - Task type
    - Dataset type
    - Number of layers

1. MLP (Does not have a explicit implementation)
Global Parameters:
    - None

2. CNN
Global Parameters:
    - CNN stage num

3. GCN
Global Parameters:
    - None

4. GAT
Global Parameters:
    - None

5. Transformer
Global Parameters:
    - Number of encoders

6. KAN
Global Parameters:
    - None

7. Recursive
Global Parameters:
    - Structure #freqs
    - Index #freqs
'''

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from enum import Enum
from ..layer import LAYER_CLS_DICT, layer_retrieve_shapes, \
    layer_retrieve_required_arange_size, layer_encode_index, \
    layer_get_params_initial_statistic, layer_apply_params_1i
from ..layer.layer import LayerUtils, IndexType
from optimization import jit_script

# Model Type
class ModelType(Enum):
    '''
    Model Type Enum
    '''
    MLP = 0
    CNN = 1
    GCN = 2
    GAT = 3
    TRANSFORMER = 4
    KAN = 5
    RECURSIVE = 6

# Task Type
class TaskType(Enum):
    '''
    Task Type Enum
    '''
    IMAGE_CLASSIFICATION = 0
    GRAPH_NODE_CLASSIFICATION = 1
    TEXT_CLASSIFICATION = 2
    FORMULA_REGRESSION = 3
    RECURSIVE = 4

# Dataset Type
class DatasetType(Enum):
    '''
    Dataset Type Enum
    '''
    # --- Image Classification ---
    MNIST = 0
    CIFAR10 = 1
    # --- Graph Node Classification ---
    CITESEER = 2
    CORA = 3
    PUBMED = 4
    # --- Text Classification ---
    AG_NEWS = 5
    IMDB = 6
    # --- Formula Regression ---
    # Special functions used in the KAN paper
    SPECIAL_ELLIPJ = 7
    SPECIAL_ELLIPKINC = 8
    SPECIAL_ELLIPEINC = 9
    SPECIAL_JV = 10
    SPECIAL_YV = 11
    SPECIAL_KV = 12
    SPECIAL_IV = 13
    SPECIAL_LPMV0 = 14
    SPECIAL_LPMV1 = 15
    SPECIAL_LPMV2 = 16
    SPECIAL_SPH_HARM01 = 17
    SPECIAL_SPH_HARM11 = 18
    SPECIAL_SPH_HARM02 = 19
    SPECIAL_SPH_HARM12 = 20
    SPECIAL_SPH_HARM22 = 21
    # --- Recursive ---
    RECURSIVE_IMAGE_CLASSIFICATION = 22
    RECURSIVE_GRAPH_NODE_CLASSIFICATION = 23
    RECURSIVE_TEXT_CLASSIFICATION = 24
    RECURSIVE_FORMULA_REGRESSION = 25
    # --- A 3D version of MNIST for ablation ---
    MNIST_3D = 26

# Input features (input size) for each dataset
FEATURES_DICT: Dict[int, int] = {
    # Image Classification
    DatasetType.MNIST.value: 28 * 28 * 1,
    DatasetType.MNIST_3D.value: 28 * 28 * 3,
    DatasetType.CIFAR10.value: 32 * 32 * 3,
    # Graph Node Classification
    DatasetType.CORA.value: 1433,
    DatasetType.CITESEER.value: 3703,
    DatasetType.PUBMED.value: 500,
    # Formula Regression
    DatasetType.SPECIAL_ELLIPJ.value: 2,
    DatasetType.SPECIAL_ELLIPKINC.value: 2,
    DatasetType.SPECIAL_ELLIPEINC.value: 2,
    DatasetType.SPECIAL_JV.value: 2,
    DatasetType.SPECIAL_YV.value: 2,
    DatasetType.SPECIAL_KV.value: 2,
    DatasetType.SPECIAL_IV.value: 2,
    DatasetType.SPECIAL_LPMV0.value: 2,
    DatasetType.SPECIAL_LPMV1.value: 2,
    DatasetType.SPECIAL_LPMV2.value: 2,
    DatasetType.SPECIAL_SPH_HARM01.value: 2,
    DatasetType.SPECIAL_SPH_HARM11.value: 2,
    DatasetType.SPECIAL_SPH_HARM02.value: 2,
    DatasetType.SPECIAL_SPH_HARM12.value: 2,
    DatasetType.SPECIAL_SPH_HARM22.value: 2,
}
# Output classes (output size) for each dataset
OUTPUTS_DICT: Dict[int, int] = {
    # Image Classification
    DatasetType.MNIST.value: 10,
    DatasetType.MNIST_3D.value: 10,
    DatasetType.CIFAR10.value: 10,
    # Graph Node Classification
    DatasetType.CORA.value: 7,
    DatasetType.CITESEER.value: 6,
    DatasetType.PUBMED.value: 3,
    # Text Classification
    DatasetType.AG_NEWS.value: 4,
    DatasetType.IMDB.value: 2,
    # Formula Regression
    DatasetType.SPECIAL_ELLIPJ.value: 4,
    DatasetType.SPECIAL_ELLIPKINC.value: 1,
    DatasetType.SPECIAL_ELLIPEINC.value: 1,
    DatasetType.SPECIAL_JV.value: 1,
    DatasetType.SPECIAL_YV.value: 1,
    DatasetType.SPECIAL_KV.value: 1,
    DatasetType.SPECIAL_IV.value: 1,
    DatasetType.SPECIAL_LPMV0.value: 1,
    DatasetType.SPECIAL_LPMV1.value: 1,
    DatasetType.SPECIAL_LPMV2.value: 1,
    DatasetType.SPECIAL_SPH_HARM01.value: 1,
    DatasetType.SPECIAL_SPH_HARM11.value: 1,
    DatasetType.SPECIAL_SPH_HARM02.value: 1,
    DatasetType.SPECIAL_SPH_HARM12.value: 1,
    DatasetType.SPECIAL_SPH_HARM22.value: 1,
}

@jit_script
def _get_model_mode() -> Dict[str, float]:
    '''
    Get a model mode dictionary.
    '''
    model_mode: Dict[str, float] = {}
    return model_mode

@jit_script
def _encode_model_structure(structures_update: Optional[Dict[str, float]] = None) -> List[float]:
    '''
    Encode the structure of the model to a list of float values.
    The default value is set to 0.0 for all the parameters.
    '''
    structures: Dict[str, float] = {
        # --- Shared Parameters ---
        # Model type
        'model_type': 0.0,
        # Task type
        'task_type': 0.0,
        # Dataset type
        'dataset_type': 0.0,
        # Number of layers
        'num_layers': 0.0,
        # --- MLP Parameters ---
        # None
        # --- CNN Parameters ---
        # CNN stage num
        'cnn_stage_num': 0.0,
        # --- GCN Parameters ---
        # None
        # --- GAT Parameters ---
        # None
        # --- Transformer Parameters ---
        # Number of encoder layers
        'num_encoders': 0.0,
        # --- KAN Parameters ---
        # None
        # --- Recursive Parameters ---
        # Structure #freqs
        'num_structure_freqs': 0.0,
        # Index #freqs
        'num_index_freqs': 0.0
    }
    # Check if the updating values are legitimate and update the structure
    if structures_update is not None:
        for updated_key in structures_update:
            if updated_key not in structures:
                raise ValueError(f"Invalid attribute: {updated_key}")
        # Update the structures with the given values
        structures.update(structures_update)
    return list(structures.values())

@jit_script
def _generate_mode(param_specified: Dict[str, float],
                  param_sampled: Dict[str, List[float]]
                  ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    '''
    Generate the model strcutral details, global and local (layer-wise).
    
    param_specified: specified parameters (structure) for the model.
    param_sampled: distribution of the parameters for sampling.
    '''
    # Generate the mode of the global structure
    global_mode: Dict[str, float] = {}
    # Generate the mode of the local structure
    local_mode: List[Dict[str, float]] = []
    return global_mode, local_mode

@jit_script
def _generate_mode_for_size(param_specified: Dict[str, float]
                           ) -> List[Dict[str, float]]:
    '''
    Only generate mode that is for model size calculation, which means that 
        when this flag is true, only the size related attributes in this mode 
        is filled. Under this mode, all attributes are filled through 
        param_specified, and param_sampled should be left empty. Notice that 
        the model size deduced through the mode under this flag=true could be 
        slightly larger than expected given that some attributes like bias_type 
        are always set to true in this case.
    param_specified: specified parameters (structure) for the model, which should
        reflect the case with maximum #parameters.
    Returns:
        - local_mode: local structure of the model.
    '''
    # Generate the mode of the local structure
    local_mode: List[Dict[str, float]] = []
    return local_mode      
  
@jit_script
def _retrieve_max_memory_size(param_specified: Dict[str, float],
                             param_sampled: Dict[str, List[float]]
                             ) -> Tuple[int, int, int, int, int]:
    '''
    Retrieve the maximum should-be pre-allocated memory size for:
        structure encoding:
            - global structure encoding
            - local structure encoding
        index encoding:
            - index encoding
            - shared element buffer
            - arange tensor
    param_specified: specified parameters for the model.
    param_sampled: distribution of the parameters for sampling.
    Returns:
        structure encoding:
            - global_structure_size: size of the memory for global structure encoding.
            - local_structure_size: size of the memory for local structure encoding.
        index encoding:
            - encode_memory_size: size of the memory for encoding.
            - shared_element_buffer_size: size of the memory for shared elements.
            - arange_tensor_size: size of the memory for arange tensor.
    '''
    # Structure encoding
    global_structure_size: int = 0
    local_structure_size: int = 0
    # Index encoding
    encode_memory_size: int = 0
    shared_element_buffer_size: int = 0
    arange_tensor_size: int = 0
    return global_structure_size, local_structure_size, \
        encode_memory_size, shared_element_buffer_size, arange_tensor_size

@jit_script
def _retrieve_shapes(local_mode: List[Dict[str, float]]
                    ) -> Tuple[List[List[int]], List[List[List[int]]]]:
    '''
    Retrieve the lengths and shapes of the parameters for each layer in the model.
    local_mode: local structure of the model.
    '''
    # Collect the lengths and shapes of the parameters for each layer
    layers_lens_model: List[List[int]] = []
    layers_param_shapes_model: List[List[List[int]]] = []
    for layer_idx in range(len(local_mode)):
        layer_type: int = int(local_mode[layer_idx]['layer_type'])
        layer_lens, layer_param_shapes = layer_retrieve_shapes(
            layer_type=layer_type,
            layer_structure=local_mode[layer_idx]
        )        
        layers_lens_model.append(layer_lens)
        layers_param_shapes_model.append(layer_param_shapes)
    return layers_lens_model, layers_param_shapes_model

@jit_script
def _retrieve_required_arange_size(local_mode: List[Dict[str, float]]) -> int:
    '''
    Retrieve the size of the arange tensor required for encoding the model.
    local_mode: local structure of the model.
    '''
    arange_size: int = 0
    for layer_idx in range(len(local_mode)):
        layer_type: int = int(local_mode[layer_idx]['layer_type'])
        arange_size_layer: int = layer_retrieve_required_arange_size(
            layer_type=layer_type,
            layer_structure=local_mode[layer_idx]
        )        
        arange_size = max(arange_size, arange_size_layer)
    return arange_size

@jit_script
def _update_max_memory_size(global_structure_size: int,
                           local_structure_size: int,
                           encode_memory_size: int,
                           shared_element_buffer_size: int,
                           arange_tensor_size: int,
                           local_mode: List[Dict[str, float]]
                           ) -> Tuple[int, int, int, int, int]:
    '''
    Update the maximum should-be pre-allocated memory size given the local structure.
    '''
    # Implement the calculation for the maximum memory size and update the values
    # Structure encoding
    global_structure_size = max(global_structure_size, 
        len(_encode_model_structure(None)))
    local_structure_size = max(local_structure_size,
        len(LayerUtils.encode_layer_structure(None)) * len(local_mode))
    # Index encoding
    layers_lens_model, _ = _retrieve_shapes(local_mode)
    # Index attributes
    shared_attributes_num: int = IndexType.get_shared_length()
    attributes_num: int = IndexType.get_length()
    layers_lens_model_sum = 0
    for layer_lens in layers_lens_model:
        layers_lens_model_sum += sum(layer_lens)
    encode_memory_size = max(encode_memory_size, 
        layers_lens_model_sum * attributes_num)
    shared_element_buffer_size = max(shared_element_buffer_size,
        len(local_mode) * shared_attributes_num)
    arange_tensor_size = max(arange_tensor_size,
        _retrieve_required_arange_size(local_mode))
    
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
    # Default values for the min and max values are set to 0.0 for structure
    # Global structure minmax
    global_structure_minmax: torch.Tensor = torch.zeros(2, 
        len(_encode_model_structure(None)), dtype=torch.float)
    # Local structure minmax
    local_structure_minmax: torch.Tensor = torch.zeros(2, 
        len(LayerUtils.encode_layer_structure(None)), dtype=torch.float)
    # Index encoding minmax, default values are set to -1.0
    index_encoding_minmax: torch.Tensor = torch.empty(2, 
        IndexType.get_length(), dtype=torch.float)    
    index_encoding_minmax.fill_(-1.0)
    return global_structure_minmax, local_structure_minmax, index_encoding_minmax

@jit_script
def _sample_int_param_no_raise(param_specified: Dict[str, float],
                               param_sampled: Dict[str, List[float]],
                               param_name: str,
                               sample_type: str = 'minmax',
                               multiplier: int = 1,
                               divisable_by: int = 0,
                               divisor_of: int = 0,
                               check_idx: bool = False,
                               attribute_idx: int = -1
                               ) -> Tuple[int, str]:
    '''
    Sample an integer parameter from the specified and sampled parameters. No raise version
    param_specified: specified parameters for this parameter.
    param_sampled: distribution of the parameters for sampling.
    param_name: name of the parameter.
    sample_type: type of sampling (minmax, multinomial, min, max).
    multiplier: multiplier of sample value to be multiplied with. Only for minmax.
    divisable_by: the value of the sampled param after factoring should 
        be divisable by this value. Only for minmax.
    divisor_of: the sampled value of this function should be a divisor of this value.
    check_idx: if True, check if the f"param_name_{attribute_idx}" is in param_specified first.
    attribute_idx: layer idx or stage idx or encoder idx, etc. to set value for the param specifically.

    Returns:
        - param: sampled parameter value. -1 if no valid value is found.
        - error_msg: error message if no valid value is found, otherwise empty string.
    '''
    param: int = 0
    error_msg: str = ''
    param_idx_name: str = f"{param_name}_{attribute_idx}"
    if check_idx and (param_idx_name in param_specified) and (attribute_idx != -1):
        # First check if the param_idx_name is in param_specified.
        #   If it is, then use the value of param_idx_name
        param = int(param_specified[param_idx_name])
        if sample_type in ['minmax', 'min', 'max']:
            # Notice that the multiplier would not be applied here, but the divisable_by
            #   would be checked
            if divisable_by > 1 and param % divisable_by != 0:
                error_msg = f"Invalid divisable multiplier: {divisable_by}"
                return -1, error_msg
            # Check if the divisor_of is valid
            if divisor_of > 0 and divisor_of % param != 0:
                error_msg = f"Invalid divisor: {divisor_of} is not divisible by {param}"
                return -1, error_msg
    elif param_name in param_specified:
        param = int(param_specified[param_name])
        if sample_type in ['minmax', 'min', 'max']:
            param = param * multiplier
            if divisable_by > 1 and param % divisable_by != 0:
                error_msg = f"Invalid divisable multiplier: {divisable_by}"
                return -1, error_msg
            # Check if the divisor_of is valid
            if divisor_of > 0 and divisor_of % param != 0:
                error_msg = f"Invalid divisor: {divisor_of} is not divisible by {param}"
                return -1, error_msg
    else:
        if sample_type in ['minmax', 'min', 'max']:
            param_range: List[float] = param_sampled[f"{param_name}_range"]
            param_min: int = int(param_range[0])
            param_max: int = int(param_range[1])                
            if sample_type == 'min':                    
                param_max = param_min
            elif sample_type == 'max':
                param_min = param_max
            if divisable_by > 1:
                lower_bound: int = (param_min * multiplier + divisable_by - 1) // \
                    divisable_by * divisable_by
                upper_bound: int = (param_max * multiplier) // divisable_by * divisable_by
                # Check if the range is valid
                if lower_bound > upper_bound:
                    error_msg = f"Invalid range: {lower_bound} > {upper_bound}"
                    return -1, error_msg
                num_divisable: int = (upper_bound - lower_bound) // divisable_by + 1
                if sample_type == 'minmax':
                    # Check if the divisor_of is valid
                    if divisor_of > 0:
                        # Find all the divisable values in the range and sample one
                        divisors: List[int] = []
                        for i in range(num_divisable):
                            value: int = lower_bound + i * divisable_by
                            if divisor_of % value == 0:
                                divisors.append(value)
                        # Randomly sample from the divisors
                        if len(divisors) > 0:
                            # Convert divisors to a PyTorch tensor
                            divisors_tensor = torch.tensor(divisors, dtype=torch.long)
                            # Uniformly sample a random index and return the corresponding divisor
                            random_idx = torch.randint(low=0, high=len(divisors_tensor), size=(1,))
                            param = divisors_tensor[random_idx].item()
                        else:
                            error_msg = f"No valid divisors found in the range {lower_bound} to {upper_bound} for divisor {divisor_of}"
                            return -1, error_msg
                    else:
                        # Sample a random value in the range
                        param = lower_bound + torch.randint(num_divisable, (1,)).item() * divisable_by
                else:
                    param = lower_bound
                    # Check if the divisor_of is valid
                    if divisor_of > 0:
                        if divisor_of % lower_bound != 0:
                            error_msg = f"Invalid divisor: {divisor_of} is not divisible by {lower_bound}"
                            return -1, error_msg
            else:
                if sample_type == 'minmax':
                    # Check if the divisor_of is valid
                    if divisor_of > 0:
                        # Find all the divisable values in the range and sample one
                        divisors: List[int] = []
                        for i in range(param_min * multiplier, param_max * multiplier + 1):
                            if divisor_of % i == 0:
                                divisors.append(i)
                        # Randomly sample from the divisors
                        if len(divisors) > 0:
                            # Convert divisors to a PyTorch tensor
                            divisors_tensor = torch.tensor(divisors, dtype=torch.long)
                            # Uniformly sample a random index and return the corresponding divisor
                            random_idx = torch.randint(low=0, high=len(divisors_tensor), size=(1,))
                            param = divisors_tensor[random_idx].item()
                        else:
                            error_msg = f"No valid divisors found in the range {param_min * multiplier} to {param_max * multiplier} for divisor {divisor_of}"
                            return -1, error_msg
                    else:
                        param = torch.randint(param_min * multiplier,
                                                param_max * multiplier + 1,
                                                (1,)).item()
                else:
                    param = param_min
                    # Check if the divisor_of is valid
                    if divisor_of > 0:
                        if divisor_of % param_min != 0:
                            error_msg = f"Invalid divisor: {divisor_of} is not divisible by {param_min}"
                            return -1, error_msg
        elif sample_type == 'multinomial':
            param = torch.multinomial(torch.tensor(param_sampled[f"{param_name}_probs"]),
                                        1, replacement=True).item()
        else:
            error_msg = f"Invalid sample type: {sample_type}"
            return -1, error_msg
    return param, error_msg


@jit_script
def _sample_int_param(param_specified: Dict[str, float],
                      param_sampled: Dict[str, List[float]],
                      param_name: str,
                      sample_type: str = 'minmax',
                      multiplier: int = 1,
                      divisable_by: int = 0,
                      divisor_of: int = 0,
                      check_idx: bool = False,
                      attribute_idx: int = -1
                      ) -> int:
    '''
    Sample an integer parameter from the specified and sampled parameters.
    param_specified: specified parameters for this parameter.
    param_sampled: distribution of the parameters for sampling.
    param_name: name of the parameter.
    sample_type: type of sampling (minmax, multinomial, min, max).
    multiplier: multiplier of sample value to be multiplied with. Only for minmax.
    divisable_by: the value of the sampled param after factoring should 
        be divisable by this value. Only for minmax.
    divisor_of: the sampled value of this function should be a divisor of this value.
    check_idx: if True, check if the f"param_name_{attribute_idx}" is in param_specified first.
    attribute_idx: layer idx or stage idx or encoder idx, etc. to set value for the param specifically.
    '''
    param, error_msg = _sample_int_param_no_raise(
        param_specified=param_specified,
        param_sampled=param_sampled,
        param_name=param_name,
        sample_type=sample_type,
        multiplier=multiplier,
        divisable_by=divisable_by,
        divisor_of=divisor_of,
        check_idx=check_idx,
        attribute_idx=attribute_idx
    )
    if param == -1:
        # If the param is -1, it means that no valid value is found, raise an error
        raise ValueError(error_msg)
    return param

@jit_script
def _sample_odd_int_param(param_specified: Dict[str, float],
                         param_sampled: Dict[str, List[float]],
                         param_name: str
                         ) -> int:
    '''
    Sample an odd integer parameter from the specified and sampled parameters.
    '''
    param: int = 0
    if param_name in param_specified:
        # Check if the specified parameter is odd
        param = int(param_specified[param_name])
        if param % 2 == 0:
            raise ValueError(f"Invalid parameter: {param}")
    else:
        # Sample an odd integer
        param_range: List[float] = param_sampled[f"{param_name}_range"]
        param_min: int = int(param_range[0])
        param_max: int = int(param_range[1])
        num_divisable: int = (param_max - param_min) // 2 + 1
        param = param_min + 2 * torch.randint(num_divisable, (1,)).item()
    return param

@jit_script
def _sample_float_param(param_specified: Dict[str, float],
                       param_sampled: Dict[str, List[float]],
                       param_name: str,
                       sample_type: str = 'minmax'
                       ) -> float:
    '''
    Sample a float parameter from the specified and sampled parameters.
    '''
    param: float = 0.0
    if param_name in param_specified:
        param = param_specified[param_name]
    else:
        param_range: List[float] = param_sampled[f"{param_name}_range"]
        param_min: float = param_range[0]
        param_max: float = param_range[1]
        if sample_type == 'minmax':
            param = torch.rand(1).item() * (param_max - param_min) + param_min
        elif sample_type == 'min':
            param = param_min
        elif sample_type == 'max':
            param = param_max
        else:
            raise ValueError(f"Invalid sample type: {sample_type}")
    return param

@jit_script
def _encode_structure(global_mode: Dict[str, float],
                     local_mode: List[Dict[str, float]],
                     preallocated_global_memory: torch.Tensor,
                     preallocated_local_memory: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Encode the structure of the model (global) and structures of its layers (local).
    '''
    # Encode the global structure
    global_mode_list: List[float] = _encode_model_structure(global_mode)
    # Narrow the preallocated memory for global structure encoding with shape (global_dim,)
    global_structure: torch.Tensor = preallocated_global_memory.narrow(0, 0, 
        len(global_mode_list)).copy_(torch.tensor(global_mode_list))
    # Encode the local structure
    local_structure_lists: List[List[float]] = [
        LayerUtils.encode_layer_structure(local_mode_layer)
        for local_mode_layer in local_mode
    ]
    # Local structures
    local_structures: torch.Tensor = torch.tensor(local_structure_lists)
    # Narrow the preallocated memory for local structure encoding and move to cuda,
    # and reshape to(1, layer_num, local_dim)
    local_structures: torch.Tensor = preallocated_local_memory.narrow(0, 0,
        local_structures.numel()).view_as(local_structures).copy_(local_structures).unsqueeze(0)
    return global_structure, local_structures
  
@jit_script
def _encode_index(local_mode: List[Dict[str, float]],
                 layers_lens: List[List[int]],
                 preallocated_encode_memory: torch.Tensor, 
                 preallocated_shared_element_buffer: torch.Tensor,
                 arange_tensor: torch.Tensor
                 ) -> torch.Tensor:
    '''
    Encode the index of the weights for all the layers in the model.
    local_mode: local structure of the model.
    layers_lens: sizes of the elements of the layers.
    preallocated_encode_memory: preallocated memory for encoding.
    preallocated_shared_element_buffer: preallocated memory for shared elements.
    arange_tensor: a pre-allocated memory with arange tensor.
    '''
    # Collect the size of the elements for slicing the memory.
    # Including the whole size of a layer and the size of each element.
    slice_memory_size_layerwise: List[int] = []
    slice_memory_size_elementwise: List[int] = []
    # Collect the shared attributes across layer elements
    basic_shared_element_list: List[List[float]] = []
    # Generate size of memory that is needed for each layer or element
    for layer_idx in range(len(local_mode)):
        # Get the sizes of the elements for the layer
        layer_lens: List[int] = layers_lens[layer_idx]
        # Collect the sizes
        slice_memory_size_layerwise.append(sum(layer_lens))
        slice_memory_size_elementwise.extend(layer_lens)
        # Collect the shared attributes (layer index and layer type)
        basic_shared_element_list.append([
            local_mode[layer_idx]['layer_idx'],
            local_mode[layer_idx]['layer_type']
        ])
    # Narrow the preallocated memory for encoding
    index_type_num: int = IndexType.get_length()
    encode_memory: torch.Tensor = preallocated_encode_memory.narrow(0, 0, 
        sum(slice_memory_size_layerwise) * index_type_num).view(-1, index_type_num)
    # Set default values for the memory as -1
    encode_memory.fill_(-1)
    
    # Slice out the shared attributes layerwise
    shared_attributes_num: int = IndexType.get_shared_length()
    # shared_attributes_num: int = ModelUtils.INDEX_TYPE_SHARED_LENGTH
    sliced_shared_memories: List[torch.Tensor] = torch.split(encode_memory[:, :shared_attributes_num],
                                                     slice_memory_size_layerwise, dim=0)

    # Collect results
    sliced_unique_memories_dict: Dict[str, List[torch.Tensor]] = {
        attr_name: torch.split(encode_memory[:, attr_idx],
                              slice_memory_size_elementwise, dim=0)
        for attr_name, attr_idx in IndexType.get_unique_dict().items()
    }

    # Compute layer boundaries
    layer_boundaries: List[int] = [0]
    total: int = 0
    for layer_lens in layers_lens:
        total += len(layer_lens)
        layer_boundaries.append(total)
    # Group the sliced unique memories by layer
    sliced_unique_memories_grouped_dict: Dict[str, List[List[torch.Tensor]]] = {
        attr_name: [sliced_unique_memories_dict[attr_name][start:end]
                    for start, end in zip(layer_boundaries[:-1], layer_boundaries[1:])]
        for attr_name in sliced_unique_memories_dict.keys()
    }
    # Reorganize the unique memories dict
    sliced_unique_memory_dicts: List[Dict[str, List[torch.Tensor]]] = [
        {attr_name: sliced_unique_memories_grouped_dict[attr_name][layer_idx]
            for attr_name in sliced_unique_memories_grouped_dict.keys()}
        for layer_idx in range(len(local_mode))
    ]
    
    # Narrow the preallocated memory for shared elements
    shared_element_buffer: torch.Tensor = preallocated_shared_element_buffer.narrow(0, 0,
        len(basic_shared_element_list) * shared_attributes_num).view(-1, shared_attributes_num)
    # Copy the shared attributes to the shared element buffer
    shared_element_buffer.copy_(torch.tensor(basic_shared_element_list, dtype=torch.float))
    # Slice the shared attributes layerwise
    shared_elements: List[torch.Tensor] = torch.split(
        shared_element_buffer,
        1, 
        dim=0)
    
    # Encode idx for each layer
    for layer_idx in range(len(local_mode)):
        layer_type: int = int(local_mode[layer_idx]['layer_type'])
        # Call the index encoding function of the layer
        layer_encode_index(
                       layer_type=layer_type,
                       layer_structure=local_mode[layer_idx],
                       shared_element=shared_elements[layer_idx],
                       shared_memory=sliced_shared_memories[layer_idx],
                       unique_memory=sliced_unique_memory_dicts[layer_idx],
                       arange_tensor=arange_tensor)
                           
    return encode_memory

@jit_script
def _retrieve_weights(weights: torch.Tensor,
                     layers_lens: List[List[int]],
                     ) -> List[List[torch.Tensor]]:
    '''
    Retrieve the weights tensor for each layer in the model.
    weights: the weights tensor of the model.
    layers_lens: sizes of the elements of the layers.
    '''
    # Collect the sizes of the elements for slicing the weights
    element_lens: List[int] = []
    for layer_lens in layers_lens:
        for element_len in layer_lens:
            element_lens.append(element_len)
    # Slice the weights tensor into element-wise
    sliced_weights: List[torch.Tensor] = torch.split(weights, element_lens, dim=0)
    # Recombine the sliced weights into layer-wise
    layer_boundaries: List[int] = [0]
    total: int = 0
    for layer_lens in layers_lens:
        total += len(layer_lens)
        layer_boundaries.append(total)    
    layers_params: List[List[torch.Tensor]] = [
        sliced_weights[start:end]
        for start, end in zip(layer_boundaries[:-1], layer_boundaries[1:])
    ]
    return layers_params

@jit_script
def _get_weights_initial_statistic(local_mode: List[Dict[str, float]]
                                  ) -> List[List[Tuple[float, float]]]:
    '''
    Get the initial statistics (mean, std) of the weights for each layer in the model.
    '''
    # Collect the initial statistics for the weights of each layer
    layers_stats: List[List[Tuple[float, float]]] = []
    for layer_idx in range(len(local_mode)):
        layer_type: int = int(local_mode[layer_idx]['layer_type'])
        layer_stats = layer_get_params_initial_statistic(
            layer_type=layer_type,
            layer_structure=local_mode[layer_idx]
        )
        layers_stats.append(layer_stats)
    return layers_stats

@jit_script
def _apply_weights(x: torch.Tensor, 
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
        x = layer_apply_params_1i(
            layer_type=layer_type,
            x=x,
            layer_structure=local_mode[layer_idx],
            layer_params=layers_params[layer_idx],
            layer_param_shapes=layers_param_shapes[layer_idx],
            training=training
        )
    return x

class ModelUtils:
    '''
    Model Class for encoding and forwarding
    '''
    get_model_mode = staticmethod(_get_model_mode)
    encode_model_structure = staticmethod(_encode_model_structure)
    generate_mode = staticmethod(_generate_mode)
    generate_mode_for_size = staticmethod(_generate_mode_for_size)
    retrieve_max_memory_size = staticmethod(_retrieve_max_memory_size)
    update_max_memory_size = staticmethod(_update_max_memory_size)
    retrieve_required_arange_size = staticmethod(_retrieve_required_arange_size)
    retrieve_encode_input_minmax = staticmethod(_retrieve_encode_input_minmax)
    sample_int_param_no_raise = staticmethod(_sample_int_param_no_raise)
    sample_int_param = staticmethod(_sample_int_param)
    sample_odd_int_param = staticmethod(_sample_odd_int_param)
    sample_float_param = staticmethod(_sample_float_param)
    encode_structure = staticmethod(_encode_structure)
    encode_index = staticmethod(_encode_index)
    retrieve_shapes = staticmethod(_retrieve_shapes)
    retrieve_weights = staticmethod(_retrieve_weights)
    get_weights_initial_statistic = staticmethod(_get_weights_initial_statistic)
    apply_weights = staticmethod(_apply_weights)

class Model(nn.Module):
    '''
    Model Class for encoding and forwarding
    '''
    @classmethod
    def modularize(cls, 
                   local_mode: List[Dict[str, float]],
                   layers_params: List[List[torch.Tensor]],
                   layers_param_shapes: List[List[List[int]]],      
                   ) -> nn.Module:
        '''
        Modularize the model with the given arguments.
        '''
        return cls(local_mode, layers_params, layers_param_shapes)
                
    def __init__(self, 
                 local_mode: List[Dict[str, float]],
                 layers_params: List[List[torch.Tensor]],
                 layers_param_shapes: List[List[List[int]]],
                 ) -> None:
        '''
        Initialize the model with the given parameters.
        '''
        super(Model, self).__init__()

        # Traverse the layers and save them
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer_idx in range(len(local_mode)):
            layer_type: int = int(local_mode[layer_idx]['layer_type'])
            self.layers.append(LAYER_CLS_DICT[layer_type].modularize(
                local_mode[layer_idx],
                layers_params[layer_idx],
                layers_param_shapes[layer_idx]
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward the input tensor through the model.
        '''
        for layer in self.layers:
            x = layer(x)
        return x
    