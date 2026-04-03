'''
Solving tasks via hypernetworks
'''
import torch
from typing import Dict, List, Tuple, Union, Optional
from model.model import MODEL_CLS_DICT, MODEL_UTILS_DICT
from model.model.model import Model, ModelUtils, DatasetType, TaskType, ModelType
from dataset.dataset import seed_worker
from dataset.text_dataset import tokenize, build_vocab, IMDBDataset, \
    AGNewsDataset, collate_text_batch, PAD_TOKEN_IDX
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, Subset
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data.data import BaseData
from scipy import special
import numpy as np
import math
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
from . import image_classification, graph_node_classification, \
    text_classification, formula_regression, recursive
from enum import Enum
from functools import partial
from optimization import jit_script, get_precision_ctx, get_grad_scaler
from decimal import Decimal, ROUND_HALF_UP

class TransformSubset(Dataset):
    """A subset dataset with transform support"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class InfiniteDataLoader:
    '''
    Infinite data loader for the training process
    '''
    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader
        # Preload iterator
        self._iterator = iter(data_loader)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (inputs, targets) for the next step.
        """
        # Get batch (reset iterator if exhausted)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.data_loader)
            return next(self._iterator)
        
    def __iter__(self):
        """Allows for-loop usage (though step-based is more common)."""
        while True:
            yield self.__next__()

    def reset(self) -> None:
        """
        Resets the iterator to the beginning of the dataset.
        """
        self._iterator = iter(self.data_loader)

class ModelModeDataset(Dataset):
    def __init__(self, model_modes):
        self.model_modes = model_modes

    def __len__(self):
        return len(self.model_modes)

    def __getitem__(self, idx):
        return self.model_modes[idx]
    
class ModelModeWithLabelDataset(Dataset):
    def __init__(self, model_modes):
        self.model_modes = model_modes
        self.label = torch.arange(len(model_modes))

    def __len__(self):
        return len(self.model_modes)

    def __getitem__(self, idx):
        return self.model_modes[idx], self.label[idx]

def round_float(x, precision=3):
    return float(Decimal(x).quantize(Decimal(f'1e-{precision}'), rounding=ROUND_HALF_UP))

def round_floats_in_structure(data, precision=3):
    if isinstance(data, float):
        return round_float(data, precision)
    elif isinstance(data, dict):
        return {k: round_floats_in_structure(v, precision) for k, v in data.items()}
    elif isinstance(data, list):
        return [round_floats_in_structure(item, precision) for item in data]
    elif isinstance(data, tuple):
        return tuple(round_floats_in_structure(item, precision) for item in data)
    else:
        return data
    
class ModelSetParams:
    '''
    Parameters for model sets
    '''
    def __init__(self,
                 multi_model_mode: str,
                 test_mode: Optional[str] = None,
                 loader_S: Optional[InfiniteDataLoader] = None,
                 loader_S_prime: Optional[DataLoader] = None,
                 loader_S_train: Optional[InfiniteDataLoader] = None,
                 loader_S_train_prime: Optional[DataLoader] = None,
                 loader_S_val_prime: Optional[DataLoader] = None,
                 loader_S_test_prime: Optional[DataLoader] = None
    ) -> None:
        # "fixed" or "on-fly"
        self.multi_model_mode = multi_model_mode
        # "full_set" or "hold_out"
        self.test_mode = test_mode
        # Available when "test_mode" is "full_set" 
        self.loader_S = loader_S
        self.loader_S_prime = loader_S_prime
        # Available when "test_mode" is "hold_out"
        self.loader_S_train = loader_S_train
        self.loader_S_train_prime = loader_S_train_prime
        self.loader_S_val_prime = loader_S_val_prime
        self.loader_S_test_prime = loader_S_test_prime

    def reset_data_loaders(self) -> None:
        '''
        Reset the data loaders to the beginning of the dataset.
        '''
        if self.multi_model_mode == "fixed":
            if self.test_mode == "full_set":
                self.loader_S.reset()
            elif self.test_mode == "hold_out":
                self.loader_S_train.reset()
            else:
                raise ValueError(f"Unknown test mode: {self.test_mode}")
        elif self.multi_model_mode == "on-fly":
            raise ValueError("On-the-fly sampling is not supported in this method.")
        else:
            raise ValueError(f"Unknown multi-model mode: {self.multi_model_mode}")

    def __next__(self) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Samples the next global and local model mode.
        Returns:
            global_mode: the global structure of the model.
            local_mode: the local structures of the model.
        """
        # Sample the next global and local model mode
        if self.multi_model_mode == "fixed":
            if self.test_mode == "full_set":
                return next(self.loader_S)
            elif self.test_mode == "hold_out":
                return next(self.loader_S_train)
            else:
                raise ValueError(f"Unknown test mode: {self.test_mode}")
        elif self.multi_model_mode == "on-fly":
            raise ValueError("On-the-fly sampling is not supported in this method.")
        else:
            raise ValueError(f"Unknown multi-model mode: {self.multi_model_mode}")

class TaskParams:
    '''
    Parameters for tasks
    '''
    def __init__(self,
                 param_specified: Dict[str, float],
                 param_sampled: Dict[str, List[float]],
                 loss_func: Optional[torch.nn.Module] = None,
                 train_inf_loader: Optional[InfiniteDataLoader] = None,
                 train_loader: Optional[DataLoader] = None,
                 validate: Optional[bool] = False,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 graph_dataset: Optional[Planetoid] = None,
                 test_configs: Optional[List[Dict[str, float]]] = None,
                 model_set_params: Optional[ModelSetParams] = None
                 ) -> None:
        '''
        Initialize the task parameters.
        '''
        # Task training and testing parameters
        self.param_specified = param_specified
        self.param_sampled = param_sampled
        self.test_configs = test_configs
        # Task loss function
        self.loss_func = loss_func
        # Task data loaders
        self.train_inf_loader = train_inf_loader
        self.train_loader = train_loader
        self.validate = validate
        self.val_loader = val_loader
        self.test_loader = test_loader
        # Task graph dataset
        self.graph_dataset = graph_dataset
        # Task model set parameters, only available when multi-model is activated
        self.model_set_params = model_set_params

    def reset_model_set_data_loaders(self) -> None:
        '''
        Reset the data loaders for the model set to the beginning of the dataset.
        '''
        if self.model_set_params is not None and \
                self.model_set_params.multi_model_mode == "fixed":
            self.model_set_params.reset_data_loaders()

    def __next__(self) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Samples the next global and local model mode.
        Returns:
            global_mode: the global structure of the model.
            local_mode: the local structures of the model.
        """
        if self.model_set_params is not None and \
                self.model_set_params.multi_model_mode == "fixed":
            return next(self.model_set_params)
        else:
            return None, None

class PreallocatedMemory:
    '''
    Preallocated memory for task
    '''
    def __init__(self,
                 global_structure_size: int,
                 local_structure_size: int,
                 encode_memory_size: int,
                 shared_element_buffer_size: int,
                 arange_tensor_size: int,
                 ) -> None:
        '''
        Initialize the preallocated memory.
        '''
        # Pre-allocate memory for structure encoding
        self.global_memory = torch.zeros(global_structure_size, dtype=torch.float,
                                         device='cuda').contiguous()
        self.local_memory = torch.zeros(local_structure_size, dtype=torch.float,
                                         device='cuda').contiguous()
        # Pre-allocate memory for indices encoding
        self.encode_memory = torch.zeros(encode_memory_size, dtype=torch.float, 
                                         device='cuda').contiguous()
        self.shared_element_buffer = torch.zeros(shared_element_buffer_size,
                                         dtype=torch.float, device='cuda').contiguous()
        self.arange_tensor = torch.arange(arange_tensor_size, dtype=torch.float, 
                                          device='cuda').contiguous()
        
class DatasetSplit(Enum):
    '''
    Dataset split for training, validation and testing
    '''
    TRAIN = 0
    VAL = 1
    TEST = 2

@jit_script
def cal_layer_mean_stds(layer_params: List[torch.Tensor]
                        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    '''
    Calculate the initialization loss for a single layer.
    layer_params: the parameters for this layer.
    target_layer_stats: the target statistics for this layer.
    Returns:
        mean_stds_layer: the mean and std of the parameters in the layer.
    '''
    # Calculate the statistics of the weights, notice that torch.std_mean returns 
    #   std first and then mean, but we need mean first and then std, whose order
    #   is required to be reversed.
    mean_stds_layer: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for param in layer_params:
        if param.numel() > 0:
            mean_stds_layer.append(torch.std_mean(param, unbiased=False)[::-1])
        else:
            mean_stds_layer.append((torch.empty(0), torch.empty(0)))

    return mean_stds_layer
    
class Task:
    '''
    Solving tasks via hypernetworks
    '''
    def __init__(self,
                 task_params: TaskParams,
                 ) -> None:
        '''
        Initialize the task.
        '''
        # Task parameters
        self.task_params: TaskParams = task_params
        # Pre-allocate memory for structure and index encoding
        self._preallocated_memory: Optional[PreallocatedMemory] = None

    @staticmethod
    def calculate_max_memory_size(task_params: TaskParams
                                  ) -> Tuple[int, int, int, int, int]:
        '''
        Calculate the maximum memory size for structure and index encoding.  

        Returns:
            global_structure_size: the maximum memory size for global structure.
            local_structure_size: the maximum memory size for local structures.
            encode_memory_size: the maximum memory size for encoding.
            shared_element_buffer_size: the maximum memory size for shared elements.
            arange_tensor_size: the maximum memory size for arange tensor.          
        '''
        # Retrieve the model type
        model_type: int = int(task_params.param_specified["model_type"])
        model_utils: type = MODEL_UTILS_DICT[model_type]
        # Calculate the maximum memory size for structure and index encoding
        global_structure_size, local_structure_size, encode_memory_size, shared_element_buffer_size, arange_tensor_size = \
            model_utils.retrieve_max_memory_size(task_params.param_specified, task_params.param_sampled)
        # Try calculating the maximum memory size if test params are provided
        if task_params.test_configs is not None:
            for test_config in task_params.test_configs:
                global_structure_size_test, local_structure_size_test, encode_memory_size_test, shared_element_buffer_size_test, \
                    arange_tensor_size_test = model_utils.retrieve_max_memory_size(test_config, {})
                # Update the maximum memory size if test params are larger
                global_structure_size = max(global_structure_size, global_structure_size_test)
                local_structure_size = max(local_structure_size, local_structure_size_test)
                encode_memory_size = max(encode_memory_size, encode_memory_size_test)
                shared_element_buffer_size = max(shared_element_buffer_size, shared_element_buffer_size_test)
                arange_tensor_size = max(arange_tensor_size, arange_tensor_size_test)
        return global_structure_size, local_structure_size, encode_memory_size, shared_element_buffer_size, arange_tensor_size

    @property
    def preallocated_memory(self) -> PreallocatedMemory:
        '''
        Preallocated memory for structure and index encoding
        '''
        # Pre-allocate memory for structure and index encoding    
        if self._preallocated_memory is None:
            # Calculate the maximum memory size for structure and index encoding
            global_structure_size, local_structure_size, encode_memory_size, shared_element_buffer_size, arange_tensor_size = \
                Task.calculate_max_memory_size(self.task_params)    
            self._preallocated_memory = PreallocatedMemory(
                global_structure_size=global_structure_size,
                local_structure_size=local_structure_size,
                encode_memory_size=encode_memory_size,
                shared_element_buffer_size=shared_element_buffer_size,
                arange_tensor_size=arange_tensor_size
            )
            print(f"[Info] Preallocating memory: ")
            print(f"    global_structure_size: {global_structure_size}")
            print(f"    local_structure_size: {local_structure_size}")
            print(f"    encode_memory_size: {encode_memory_size}")
            print(f"    shared_element_buffer_size: {shared_element_buffer_size}")
            print(f"    arange_tensor_size: {arange_tensor_size}")            
        return self._preallocated_memory

    @staticmethod
    def retrieve_encode_input_minmax(task_params: TaskParams
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Retrieve the min and max values for the input for encoding, including
            global and local structures, and the index encoding. Notice that the minmax
            should be only calculated based on the training params in case of structual leaking.
            
        Returns:
            - global_structure_minmax: min and max values for the global structure.
            - local_structure_minmax: min and max values for the local structure.
            - index_encoding_minmax: min and max values for the index encoding.
        '''
        # Retrieve the model type
        model_type: int = int(task_params.param_specified["model_type"])
        model_utils: type = MODEL_UTILS_DICT[model_type]
        # Calculate the min and max values for the input for encoding
        global_structure_minmax, local_structure_minmax, index_encoding_minmax = \
            model_utils.retrieve_encode_input_minmax(task_params.param_specified,
                                                    task_params.param_sampled)
        return global_structure_minmax, local_structure_minmax, index_encoding_minmax

    def calculate_encode_input_minmax(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Calculate the min and max values for the input for encoding, including
            global and local structures, and the index encoding. Notice that the minmax
            should be only calculated based on the training params in case of structual leaking.
            
        Returns:
            - global_structure_minmax: min and max values for the global structure.
            - local_structure_minmax: min and max values for the local structure.
            - index_encoding_minmax: min and max values for the index encoding.
        '''
        return Task.retrieve_encode_input_minmax(self.task_params)  

    @staticmethod
    def model_params_generate(hypernet: torch.nn.Module,
                              task_params: TaskParams,
                              test_config_idx: Optional[int],
                              preallocated_memory: PreallocatedMemory,
                              global_mode: Optional[Dict[str, float]] = None,
                              local_mode: Optional[List[Dict[str, float]]] = None
                              ) -> Tuple[torch.Tensor,
                                         List[List[torch.Tensor]], List[List[List[int]]], 
                                         Dict[str, float], List[Dict[str, float]]]:
        '''
        Generate model parameters through hypernetworks.

        Args:
            hypernet: hypernetwork.
            task_params: task parameters.
            test_config_idx: index of the test configuration. None if use the training configuration.
            preallocated_memory: preallocated memory for structure and index encoding.
            global_mode: the global structure of the model. If None, it will be generated on-the-fly.
            local_mode: the local structures of the model. If None, it will be generated on-the-fly.
            
        Returns:
            weights: the vanilla weights generated by the hypernetwork.
            layers_params: the parameters for each layer.
            layers_param_shapes: the shapes of the parameters for each layer.
            global_mode: the global structure of the model.
            local_mode: the local structures of the model.
        '''
        if global_mode is None or local_mode is None:
            # Retrieve the model type
            if test_config_idx is None:
                # Use the training configuration
                param_specified: Dict[str, float] = task_params.param_specified
                param_sampled: Dict[str, List[float]] = task_params.param_sampled
            else:
                # Use the test configuration
                param_specified: Dict[str, float] = task_params.test_configs[test_config_idx]
                param_sampled: Dict[str, List[float]] = {}
            model_type: int = int(param_specified["model_type"])
            model_utils: type = MODEL_UTILS_DICT[model_type]
            # Generate model mode
            global_mode, local_mode = model_utils.generate_mode(param_specified, 
                                                                param_sampled)
        # Encode model structure
        global_structure, local_structures = ModelUtils.encode_structure(
            global_mode, 
            local_mode,
            preallocated_memory.global_memory,
            preallocated_memory.local_memory)
        # Retrieve the lengths and shapes of the parameters of the model
        layers_lens, layers_param_shapes = ModelUtils.retrieve_shapes(local_mode)
        # Encode model index
        idxes: torch.Tensor = ModelUtils.encode_index(
            local_mode,
            layers_lens,
            preallocated_memory.encode_memory,
            preallocated_memory.shared_element_buffer,
            preallocated_memory.arange_tensor)
        # Pass the model encoding to the hypernetwork to generate the weights
        weights: torch.Tensor = hypernet(global_structure=global_structure,
                                        local_structures=local_structures,
                                        idxes=idxes)
        # Retrieve the params for each layer
        layers_params: List[List[torch.Tensor]] = ModelUtils.retrieve_weights(weights,
                                                                            layers_lens)
        return weights, layers_params, layers_param_shapes, global_mode, local_mode
    
    @staticmethod
    def sample_model_modes(param_specified: Dict[str, float],
                           param_sampled: Dict[str, List[float]],
                           sample_num: int,
                           max_trials=50000
                           ) -> List[Tuple[Dict[str, float], List[Dict[str, float]]]]:
        '''
        Sample model modes from the task parameters.
        Args:
            param_specified: specified parameters for the model.
            param_sampled: sampled parameters for the model.
            sample_num: number of samples to generate.
        Returns:
            sampled_modes: a list of sampled modes, each mode is a tuple of global mode and local modes.
        '''
        # Retrieve the model type
        model_type: int = int(param_specified["model_type"])
        model_utils: type = MODEL_UTILS_DICT[model_type]
        
        seen: set = set()
        model_modes: List[Tuple[Dict[str, float], List[Dict[str, float]]]] = []
        for sample_idx in tqdm(range(max_trials), desc="Sampling model modes"):
            # Sample model mode
            global_mode: Dict[str, float]
            local_mode: List[Dict[str, float]]            
            global_mode, local_mode = model_utils.generate_mode(param_specified, 
                                                                param_sampled)
            model_mode = (global_mode, local_mode)
            # Round all floats to avoid precision issues
            model_mode = round_floats_in_structure(model_mode)
            # Serialize using fixed float precision
            canonical = json.dumps(model_mode, sort_keys=True)
            if canonical not in seen:
                seen.add(canonical)
                model_modes.append(model_mode)

            # Check if the number of samples is enough
            if len(model_modes) >= sample_num:
                break

        assert len(model_modes) == sample_num, \
            f"Failed to sample {sample_num} unique model modes, only sampled {len(model_modes)} modes."
        return model_modes

    @staticmethod
    def model_infer(x: List[torch.Tensor],
                    hypernet: torch.nn.Module,
                    task_params: TaskParams,
                    test_config_idx: Optional[int],
                    preallocated_memory: PreallocatedMemory,
                    training: bool,
                    global_mode: Optional[Dict[str, float]] = None,
                    local_mode: Optional[List[Dict[str, float]]] = None                    
                    ) -> Tuple[torch.Tensor,
                            Dict[str, float], List[Dict[str, float]],
                            List[List[torch.Tensor]]]:
        '''
        Whole process of model inference through hypernetworks.

        Args:
            x: a list of input tensors.
            hypernet: hypernetwork.
            task_params: task parameters.
            test_config_idx: index of the test configuration. None if use the training configuration.
            preallocated_memory: preallocated memory for structure and index encoding.
            training: training mode.
            global_mode: the global structure of the model. If None, it will be generated on-the-fly.
            local_mode: the local structures of the model. If None, it will be generated on-the-fly.
        
        Returns:
            x: the output tensor.
            global_mode: the global structure of the model.
            local_mode: the local structures of the model.
            layers_params: the parameters for each layer.
        '''
        # Retrieve the model type
        if test_config_idx is None:
            # Use the training configuration
            param_specified: Dict[str, float] = task_params.param_specified
        else:
            # Use the test configuration
            param_specified: Dict[str, float] = task_params.test_configs[test_config_idx]
        model_type: int = int(param_specified["model_type"])
        model_utils: type = MODEL_UTILS_DICT[model_type]
        # Generate model params
        weights, layers_params, layers_param_shapes, global_mode, local_mode = \
            Task.model_params_generate(
                hypernet=hypernet,
                task_params=task_params,
                test_config_idx=test_config_idx,
                preallocated_memory=preallocated_memory,
                global_mode=global_mode,
                local_mode=local_mode
                )
        # Apply the weights to the input tensor
        x: torch.Tensor = model_utils.apply_weights(
            *x,
            local_mode=local_mode,
            layers_params=layers_params,
            layers_param_shapes=layers_param_shapes,
            training=training)
        return x, global_mode, local_mode, layers_params

    @staticmethod
    def model_test(hypernet: torch.nn.Module,
                   task_params: TaskParams,
                   test_config_idx: Optional[int],
                   preallocated_memory: PreallocatedMemory,
                   epoch: int,
                   epochs: int,
                   writer: Optional[SummaryWriter] = None,
                   log_str: Optional[str] = None,
                   test_split: DatasetSplit = DatasetSplit.TEST,
                   global_mode: Optional[Dict[str, float]] = None,
                   local_mode: Optional[List[Dict[str, float]]] = None                     
                   ) -> Tuple[float, Optional[float], Optional[float]]:
        '''
        Test the model on the given dataset split.

        Args:
            hypernet: hypernetwork.
            task_params: task parameters.
            test_config_idx: index of the test configuration. None if use the training configuration.
            preallocated_memory: preallocated memory for structure and index encoding.
            epoch: current epoch.
            epochs: total number of epochs.
            writer: TensorBoard writer, log nothing if None.
            log_str: string for logging.
            test_split: dataset split for testing, can be TRAIN, VAL or TEST.
            global_mode: the global structure of the model. If None, it will be generated on-the-fly.
            local_mode: the local structures of the model. If None, it will be generated on-the-fly.

        Returns:
            avg_loss: average loss of the model on the test dataset.
            avg_acc: average accuracy of the model on the test dataset, None if not applicable.
            rmse: root mean square error of the model on the test dataset, None if not applicable.
        '''
        # The task type of testing should always be the same as the task type of training in single task setting
        task_type: int = int(task_params.param_specified["task_type"])
        hypernet.eval()
        # Initialize the running statistics
        running_loss = torch.tensor(0.0, device='cuda')
        running_corrects = torch.tensor(0, device='cuda')
        total = 0
        total_classify = 0
        # Iterate over the validation / testing data
        with torch.no_grad():
            if task_type != TaskType.GRAPH_NODE_CLASSIFICATION.value:
                # Non-graph data testing
                if test_split == DatasetSplit.TRAIN:
                    loader = task_params.train_loader
                elif test_split == DatasetSplit.VAL:
                    loader = task_params.val_loader
                else:
                    loader = task_params.test_loader
                for inputs, targets in tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} [{log_str}]'):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with get_precision_ctx():
                        # Forward pass
                        if task_type == TaskType.TEXT_CLASSIFICATION.value:
                            # Get padding mask for text classification
                            padding_mask = (inputs == PAD_TOKEN_IDX)
                            x = [inputs, padding_mask]
                        else:
                            x = [inputs]
                        outputs, _, _, _ = Task.model_infer(
                            x=x,
                            hypernet=hypernet,
                            task_params=task_params,
                            test_config_idx=test_config_idx,
                            preallocated_memory=preallocated_memory,
                            training=False,
                            global_mode=global_mode,
                            local_mode=local_mode
                            )
                        # Compute the loss
                        loss = task_params.loss_func(outputs, targets)
                        # Update the running statistics
                        data_size: int = targets.size(0)
                        running_loss += loss * data_size
                        total += data_size
                        if task_type in [TaskType.IMAGE_CLASSIFICATION.value,
                                        TaskType.TEXT_CLASSIFICATION.value]:
                            # Calculate the accuracy
                            _, preds = torch.max(outputs, 1)
                            corrects = torch.sum(preds == targets)
                            running_corrects += corrects
                            total_classify += data_size
            else:
                # Graph data testing
                data: BaseData = task_params.graph_dataset[0].cuda()
                x, edge_index = data.x, data.edge_index
                with get_precision_ctx():
                    # Forward pass
                    outputs, _, _, _ = Task.model_infer(
                        x=[x, edge_index],
                        hypernet=hypernet,
                        task_params=task_params,
                        test_config_idx=test_config_idx,
                        preallocated_memory=preallocated_memory,
                        training=False,
                        global_mode=global_mode,
                        local_mode=local_mode
                        )
                    # Compute the loss
                    if test_split == DatasetSplit.TRAIN:
                        mask = data.train_mask
                    elif test_split == DatasetSplit.VAL:
                        mask = data.val_mask
                    else:
                        mask = data.test_mask
                    loss = task_params.loss_func(outputs[mask], data.y[mask])
                    # Calculate the accuracy
                    _, preds = torch.max(outputs[mask], 1)
                    corrects = torch.sum(preds == data.y[mask])
                    # Update the running statistics
                    data_size: int = data.y[mask].size(0)
                    running_loss += loss * data_size
                    running_corrects += corrects
                    total += data_size
                    total_classify += data_size
        # Log the validation statistics
        avg_loss = (running_loss / total).item()
        if writer is not None:
            writer.add_scalar(f'Loss/{log_str}', avg_loss, epoch)
            print_str = f'Epoch {epoch + 1}/{epochs} [{log_str}]: Loss: {avg_loss}'
        if total_classify > 0:
            avg_acc = (running_corrects / total_classify).item()
            if writer is not None:
                writer.add_scalar(f'Accuracy/{log_str}', avg_acc, epoch)
                print_str += f', Accuracy: {avg_acc}'
        else:
            avg_acc = None
        if task_type == TaskType.FORMULA_REGRESSION.value:
            # Log the root mean square error
            rmse = math.sqrt(avg_loss)
            if writer is not None:
                writer.add_scalar(f'RMSE/{log_str}', rmse, epoch)
                print_str += f', RMSE: {rmse}'
        else:
            rmse = None
        if writer is not None:
            print(print_str)
        return avg_loss, avg_acc, rmse

    @staticmethod
    def model_modularize(hypernet: torch.nn.Module,
                         task_params: TaskParams,
                         test_config_idx: Optional[int],
                         preallocated_memory: PreallocatedMemory,
                         global_mode: Optional[Dict[str, float]] = None,
                         local_mode: Optional[List[Dict[str, float]]] = None
                        ) -> Tuple[torch.nn.Module, Dict[str, float], List[Dict[str, float]]]:
        '''
        Modularize the model through hypernetworks.

        Args:
            hypernet: hypernetwork.
            task_params: task parameters.
            test_config_idx: index of the test configuration. None if use the training configuration.
            preallocated_memory: preallocated memory for structure and index encoding.
            global_mode: the global structure of the model. If None, it will be generated on-the-fly.
            local_mode: the local structures of the model. If None, it will be generated on-the-fly.
            
        Returns:
            network: the modularized model.
            global_mode: the global structure of the model.
            local_mode: the local structures of the model.
        '''
        # Retrieve the model type
        if test_config_idx is None:
            # Use the training configuration
            param_specified: Dict[str, float] = task_params.param_specified
        else:
            # Use the test configuration
            param_specified: Dict[str, float] = task_params.test_configs[test_config_idx]
        model_type: int = int(param_specified["model_type"])
        model_cls: type[Model] = MODEL_CLS_DICT[model_type]
        # Generate model params
        weights, layers_params, layers_param_shapes, global_mode, local_mode = \
            Task.model_params_generate(hypernet=hypernet,
                                       task_params=task_params,
                                       test_config_idx=test_config_idx,
                                       preallocated_memory=preallocated_memory,
                                       global_mode=global_mode,
                                       local_mode=local_mode)
        network: torch.nn.Module = model_cls.modularize(local_mode=local_mode,
                                                        layers_params=layers_params,
                                                        layers_param_shapes=layers_param_shapes)
        return network, global_mode, local_mode

    @staticmethod
    def set_random_seed(seed: int) -> None:
        '''
        Set random seed

        Args:
            seed: seed.
        '''
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups
        # Set specific flags for CUDA to ensure deterministic behavior
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_dataset(task_type: int,
                    dataset_type: int,
                    param_specified: Dict[str, float],
                    batch_size: int,
                    seed: int,
                    root: str = './data',
                    validate: bool = False,
                    ) -> Union[Tuple[InfiniteDataLoader, DataLoader, Optional[DataLoader], DataLoader],
                               Planetoid]:
        '''
        Get dataset.

        Args:
            task_type: task type.
            dataset_type: dataset type.
            param_specified: specified parameters for the model.
            batch_size: batch size.
            seed: seed.
            root: root directory.
            validate: whether to validate the dataset.
        
        Returns:
            Non-graph dataset: train_inf_loader, train_loader, val_loader, test_loader.
            Graph dataset: dataset.
        '''
        if task_type == TaskType.IMAGE_CLASSIFICATION.value:
            if dataset_type == DatasetType.MNIST.value or dataset_type == DatasetType.MNIST_3D.value:
                if dataset_type == DatasetType.MNIST.value:
                    train_transform  = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                else:
                    train_transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))  # Repeat mean/std for 3 channels
                    ])
                val_test_transform = train_transform 
                # Load MNIST dataset
                train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=None)
                test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=val_test_transform)
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size                
            elif dataset_type == DatasetType.CIFAR10.value:
                # Data Augmentation (Train and Test transformations)
                val_test_transform  = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                                         np.array([63.0, 62.1, 66.7]) / 255.0)
                ])
                train_transform  = transforms.Compose([
                    transforms.Pad(4, padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32),
                    val_test_transform
                ])
                # Load CIFAR-10 dataset_name
                train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
                test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=val_test_transform)
                train_size = int(0.9 * len(train_dataset))
                val_size = len(train_dataset) - train_size                
            # Split train dataset into train and validation
            if validate:
                train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], 
                                                          generator=torch.Generator().manual_seed(42 + dataset_type))
                # Apply the validation/test transform to the validation set
                val_dataset = TransformSubset(val_dataset, transform=val_test_transform)
            train_dataset = TransformSubset(train_dataset, transform=train_transform)
            # Load dataset
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                    pin_memory=True, persistent_workers=True, prefetch_factor=4, worker_init_fn=seed_worker, 
                                    generator=torch.Generator().manual_seed(seed + 1000 * dataset_type))
            if validate:
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                        pin_memory=True, persistent_workers=True, prefetch_factor=4)
            else:
                val_loader = None
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, persistent_workers=True, prefetch_factor=4)
            return InfiniteDataLoader(train_loader), train_loader, val_loader, test_loader
        elif task_type == TaskType.GRAPH_NODE_CLASSIFICATION.value:
            name_dict: Dict[int, str] = {
                DatasetType.CORA.value: 'Cora',
                DatasetType.CITESEER.value: 'CiteSeer',
                DatasetType.PUBMED.value: 'PubMed'
            }
            dataset: Planetoid = Planetoid(root=root, name=name_dict[dataset_type],
                                           transform=NormalizeFeatures())
            return dataset
        elif task_type == TaskType.TEXT_CLASSIFICATION.value:
            name_dict: Dict[int, str] = {
                DatasetType.IMDB.value: 'imdb',
                DatasetType.AG_NEWS.value: 'ag_news'
            }
            # Build vocabularies
            vocab_size: int = int(param_specified["vocab_size"])
            vocab = build_vocab(name_dict[dataset_type], root, tokenize, vocab_size)
            # Load dataset
            max_seq_len = int(param_specified["max_sequence_length"])
            if dataset_type == DatasetType.IMDB.value:
                train_dataset = IMDBDataset(root, vocab, max_seq_len, train=True)
                test_dataset = IMDBDataset(root, vocab, max_seq_len, train=False)
            elif dataset_type == DatasetType.AG_NEWS.value:
                train_dataset = AGNewsDataset(root, vocab, max_seq_len, train=True)
                test_dataset = AGNewsDataset(root, vocab, max_seq_len, train=False)
            # Split train dataset into train and validation
            if validate:    
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                                          generator=torch.Generator().manual_seed(42 + dataset_type))
            # Load dataset
            collect_fn = partial(collate_text_batch, vocab=vocab)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                    pin_memory=True, persistent_workers=False, prefetch_factor=None, worker_init_fn=seed_worker, 
                                    generator=torch.Generator().manual_seed(seed + 1000 * dataset_type), collate_fn=collect_fn)
            if validate:
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                        pin_memory=True, persistent_workers=False, prefetch_factor=None, collate_fn=collect_fn)
            else:
                val_loader = None
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                    pin_memory=True, persistent_workers=False, prefetch_factor=None, collate_fn=collect_fn)
            return InfiniteDataLoader(train_loader), train_loader, val_loader, test_loader
        elif task_type == TaskType.FORMULA_REGRESSION.value:
            # List of functions
            func_dict = {
                DatasetType.SPECIAL_ELLIPJ.value: special.ellipj,
                DatasetType.SPECIAL_ELLIPKINC.value: special.ellipkinc,
                DatasetType.SPECIAL_ELLIPEINC.value: special.ellipeinc,
                DatasetType.SPECIAL_JV.value: special.jv,
                DatasetType.SPECIAL_YV.value: special.yv,
                DatasetType.SPECIAL_KV.value: special.kv,
                DatasetType.SPECIAL_IV.value: special.iv,
                DatasetType.SPECIAL_LPMV0.value: lambda x, y: special.lpmv(0, x, y),
                DatasetType.SPECIAL_LPMV1.value: lambda x, y: special.lpmv(1, x, y),
                DatasetType.SPECIAL_LPMV2.value: lambda x, y: special.lpmv(2, x, y),
                DatasetType.SPECIAL_SPH_HARM01.value: lambda x, y: special.sph_harm(0, 1, x, y).real,
                DatasetType.SPECIAL_SPH_HARM11.value: lambda x, y: special.sph_harm(1, 1, x, y).real,
                DatasetType.SPECIAL_SPH_HARM02.value: lambda x, y: special.sph_harm(0, 2, x, y).real,
                DatasetType.SPECIAL_SPH_HARM12.value: lambda x, y: special.sph_harm(1, 2, x, y).real,
                DatasetType.SPECIAL_SPH_HARM22.value: lambda x, y: special.sph_harm(2, 2, x, y).real,
            }
            # Generate input data
            if dataset_type in [DatasetType.SPECIAL_YV.value, DatasetType.SPECIAL_KV.value]:
                # Generate input data, set to range to [epison, 1) to avoid extreme values for some functions
                epison = 1e-1
                train_x1 = np.random.rand(1000, 1)
                train_x2 = np.random.rand(1000, 1) * (1 - epison) + epison
                train_x = np.concatenate((train_x1, train_x2), axis=1)

                test_x1 = np.random.rand(1000, 1)
                test_x2 = np.random.rand(1000, 1) * (1 - epison) + epison
                test_x = np.concatenate((test_x1, test_x2), axis=1)
            else:
                # Generate input data, set to range to [0, 1)
                train_x = np.random.rand(1000, 2)
                test_x = np.random.rand(1000, 2)            
            # Generate output data
            if dataset_type == DatasetType.SPECIAL_ELLIPJ.value:
                train_y = np.stack(func_dict[dataset_type](train_x[:, 0], train_x[:, 1]), axis=1)
                test_y = np.stack(func_dict[dataset_type](test_x[:, 0], test_x[:, 1]), axis=1)
            else:
                train_y = np.expand_dims(func_dict[dataset_type](train_x[:, 0], train_x[:, 1]), axis=1)
                test_y = np.expand_dims(func_dict[dataset_type](test_x[:, 0], test_x[:, 1]), axis=1)
            # Normalize the input x to [-1, 1] range
            if dataset_type in [DatasetType.SPECIAL_YV.value, DatasetType.SPECIAL_KV.value]:
                train_x = ((train_x - np.array([[0, epison]])) / np.array([[1, 1 - epison]])) * 2 - 1
                test_x = ((test_x - np.array([[0, epison]])) / np.array([[1, 1 - epison]])) * 2 - 1
            else:
                train_x = train_x * 2 - 1
                test_x = test_x * 2 - 1
            # Create dataset
            train_dataset = TensorDataset(torch.from_numpy(train_x).float(), 
                                          torch.from_numpy(train_y).float())
            test_dataset = TensorDataset(torch.from_numpy(test_x).float(),
                                         torch.from_numpy(test_y).float())
            # Split train dataset into train and validation
            if validate:
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                                          generator=torch.Generator().manual_seed(42 + dataset_type))
            # Load dataset
            train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=4,
                                    pin_memory=True, persistent_workers=True, prefetch_factor=4, worker_init_fn=seed_worker, 
                                    generator=torch.Generator().manual_seed(seed + 1000 * dataset_type))
            if validate:
                val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=4,
                                        pin_memory=True, persistent_workers=True, prefetch_factor=4)
            else:
                val_loader = None
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4,
                                    pin_memory=True, persistent_workers=True, prefetch_factor=4)
            return InfiniteDataLoader(train_loader), train_loader, val_loader, test_loader
        raise ValueError(f"Invalid task type {task_type} or dataset type {dataset_type}")

    @staticmethod
    def generate_model_dataset(param_specified: Dict[str, float],
                               param_sampled: Dict[str, List[float]],
                               sample_full_num: int,
                               max_sample_trials: int,
                               sample_prime_num: Optional[int],
                               sample_prime_each_num: Optional[int],
                               ) -> Union[Tuple[InfiniteDataLoader, DataLoader],
                                   Tuple[InfiniteDataLoader, DataLoader, Optional[DataLoader], DataLoader]]:
        '''
        Generate model dataset, for both full set and hold out set.
        Args:
            param_specified: specified parameters for the model.
            param_sampled: sampled parameters for the model.
            sample_full_num: number of samples for full set.
            max_sample_trials: maximum number of trials for sampling of the full set.
            sample_prime_num: number of samples for prime set, available only when test_mode is "full_set".
            sample_prime_each_num: number of samples for each prime set, available only when test_mode is "hold_out".
        '''
        model_datasets = {}
        # Sample full model set
        S = Task.sample_model_modes(
            param_specified=param_specified,
            param_sampled=param_sampled,
            sample_num=sample_full_num,
            max_trials=max_sample_trials
        )
        model_datasets['S'] = S
        total_indices = list(range(len(S)))

        # Sample prime set, for full set
        indices_S_prime = total_indices[:sample_prime_num]
        model_datasets['indices_S_prime'] = indices_S_prime
            
        # Sample train and test sets, for hold out
        split = len(S) - sample_prime_each_num
        indices_S_train, indices_S_test = total_indices[:split], total_indices[split:]
        model_datasets['indices_S_train'] = indices_S_train
        model_datasets['indices_S_test'] = indices_S_test

        # Sample prime sets, for hold out
        indices_S_train_prime = indices_S_train[:sample_prime_each_num]
        model_datasets['indices_S_train_prime'] = indices_S_train_prime

        return model_datasets

    @staticmethod
    def get_model_dataset(model_datasets: Dict,
                         test_mode: str,
                         sample_prime_each_num: Optional[int],
                         validate: bool
                         ) -> Union[Tuple[InfiniteDataLoader, DataLoader],
                            Tuple[InfiniteDataLoader, DataLoader, Optional[DataLoader], DataLoader]]:
        '''
        Get model dataset
        Args:
            model_datasets: dictionary containing the model datasets.
            test_mode: test mode, "full_set" or "hold_out"
            sample_prime_each_num: number of samples for each prime set, available only when test_mode is "hold_out".
            validate: whether to validate the dataset.
        '''
        S = model_datasets['S']
        dataset_S = ModelModeDataset(S)

        if test_mode == 'full_set':
            # Sample prime set
            indices_S_prime = model_datasets['indices_S_prime']
            dataset_S_prime = Subset(dataset_S, indices_S_prime)
            # Loader for the full set
            loader_S = DataLoader(dataset_S, batch_size=1, shuffle=True)
            loader_S = InfiniteDataLoader(loader_S)
            # Loader for the prime set
            loader_S_prime = DataLoader(dataset_S_prime, batch_size=1, shuffle=False)
            return loader_S, loader_S_prime
        elif test_mode == 'hold_out':
            indices_train = model_datasets['indices_S_train']
            if validate:
                # Split the train set into train and validation sets
                split_val = len(indices_train) - sample_prime_each_num
                indices_train, indices_val = indices_train[:split_val], indices_train[split_val:]
            dataset_S_train = Subset(dataset_S, indices_train)
            # Loader for the train set
            loader_S_train = DataLoader(dataset_S_train, batch_size=1, shuffle=True)
            loader_S_train = InfiniteDataLoader(loader_S_train)

            # Sample prime sets
            if validate:
                # Just in case sample_prime_each_num is larger than split_val
                indices_train_prime = indices_train[:sample_prime_each_num]
            else:
                indices_train_prime = model_datasets['indices_S_train_prime']
            indices_test = model_datasets['indices_S_test']
            dataset_S_train_prime = Subset(dataset_S, indices_train_prime)
            dataset_S_test = Subset(dataset_S, indices_test)
            # Loader for the prime sets
            loader_S_train_prime = DataLoader(dataset_S_train_prime, batch_size=1, shuffle=False)
            loader_S_test = DataLoader(dataset_S_test, batch_size=1, shuffle=False)
            if validate:
                dataset_S_val = Subset(dataset_S, indices_val)
                # Loader for the validation prime set
                loader_S_val = DataLoader(dataset_S_val, batch_size=1, shuffle=False)
            else:
                loader_S_val = None
            return loader_S_train, loader_S_train_prime, loader_S_val, loader_S_test
        else:
            raise ValueError(f"Unknown test mode: {test_mode}")

    @staticmethod
    def get_model_with_label_dataset(model_datasets: Dict,
                         test_mode: str,
                         sample_prime_each_num: Optional[int],
                         validate: bool
                         ) -> Union[Tuple[InfiniteDataLoader, DataLoader],
                            Tuple[InfiniteDataLoader, DataLoader, Optional[DataLoader], DataLoader]]:
        '''
        Get model dataset
        Args:
            model_datasets: dictionary containing the model datasets.
            test_mode: test mode, "full_set" or "hold_out"
            sample_prime_each_num: number of samples for each prime set, available only when test_mode is "hold_out".
            validate: whether to validate the dataset.
        '''
        S = model_datasets['S']
        dataset_S = ModelModeWithLabelDataset(S)

        if test_mode == 'full_set':
            # Sample prime set
            indices_S_prime = model_datasets['indices_S_prime']
            dataset_S_prime = Subset(dataset_S, indices_S_prime)
            # Loader for the full set
            loader_S = DataLoader(dataset_S, batch_size=1, shuffle=True)
            # Loader for the prime set
            loader_S_prime = DataLoader(dataset_S_prime, batch_size=1, shuffle=False)
            return loader_S, loader_S_prime
        elif test_mode == 'hold_out':
            indices_train = model_datasets['indices_S_train']
            if validate:
                # Split the train set into train and validation sets
                split_val = int(0.8 * len(indices_train))
                indices_train, indices_val = indices_train[:split_val], indices_train[split_val:]
            dataset_S_train = Subset(dataset_S, indices_train)
            # Loader for the train set
            loader_S_train = DataLoader(dataset_S_train, batch_size=1, shuffle=True)

            # Sample prime sets
            if validate:
                # Just in case sample_prime_each_num is larger than split_val
                indices_train_prime = indices_train[:sample_prime_each_num]
            else:
                indices_train_prime = model_datasets['indices_S_train_prime']
            indices_test_prime = model_datasets['indices_S_test_prime']
            dataset_S_train_prime = Subset(dataset_S, indices_train_prime)
            dataset_S_test_prime = Subset(dataset_S, indices_test_prime)
            # Loader for the prime sets
            loader_S_train_prime = DataLoader(dataset_S_train_prime, batch_size=1, shuffle=False)
            loader_S_test_prime = DataLoader(dataset_S_test_prime, batch_size=1, shuffle=False)
            if validate:
                indices_val_prime = indices_val[:sample_prime_each_num]
                dataset_S_val_prime = Subset(dataset_S, indices_val_prime)
                # Loader for the validation prime set
                loader_S_val_prime = DataLoader(dataset_S_val_prime, batch_size=1, shuffle=False)
            else:
                loader_S_val_prime = None
            return loader_S_train, loader_S_train_prime, loader_S_val_prime, loader_S_test_prime
        else:
            raise ValueError(f"Unknown test mode: {test_mode}")

    @staticmethod
    def get_loss_function(task_type: int,
                          dataset_type: int
                          ) -> torch.nn.Module:
        '''
        Get loss function

        Args:
            task_type: task type.
            dataset_type: dataset type.
        
        Returns:
            loss_function: loss function.
        '''
        loss_dict = {
            TaskType.IMAGE_CLASSIFICATION.value: torch.nn.CrossEntropyLoss(),
            TaskType.GRAPH_NODE_CLASSIFICATION.value: torch.nn.CrossEntropyLoss(),
            TaskType.TEXT_CLASSIFICATION.value: torch.nn.CrossEntropyLoss(),
            TaskType.FORMULA_REGRESSION.value: torch.nn.MSELoss()
        }
        return loss_dict[task_type]

    @staticmethod
    def get_optimizer(hypernet: torch.nn.Module,
                      lr: float,
                      betas: Tuple[float, float] = (0.9, 0.999),
                      ) -> torch.optim.Optimizer:
        '''
        Get optimizer, always use Adam

        Args:
            hypernet: hypernetwork.
            lr: learning rate.
            betas: betas for Adam.

        Returns:
            optimizer: optimizer.
        '''
        # Use AdamW with no weight decay
        weight_decay = 0
        optimizer = AdamW(hypernet.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        return optimizer

    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer,
                      epochs: int,
                      warmup_epochs: int
                      ) -> torch.optim.lr_scheduler.SequentialLR:
        '''
        Get scheduler, always use warmup first and then cosine annealing

        Args:
            optimizer: optimizer.
            epochs: total number of epochs.
            warmup_epochs: number of warmup epochs.

        Returns:
            scheduler: scheduler.
        '''
        # Linear warmup scheduler
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / (warmup_epochs + 1))
        # Cosine annealing scheduler after warmup, use max(..., 1) to prevent division by zero when warmup_epochs = epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
        # Combine the two schedulers
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_scheduler, scheduler],
                                 milestones=[warmup_epochs])
        return scheduler
                            
    @staticmethod
    def get_hypernet_config(task_type: Optional[int] = None,
                            dataset_type: Optional[int] = None,
                            mix_status: Optional[str] = None,
                            ablation_index_fourier_n_freqs: Optional[int] = None,
                            ablation_block_num: Optional[int] = None,
                            ablation_hidden_size: Optional[int] = None,
                            ablation_index_encoding_type: Optional[str] = None,
                            ablation_index_positional_n_freqs: Optional[int] = None,
                            ablation_index_positional_sigma: Optional[float] = None
                            ) -> Dict[str, float]:
        '''
        Get hypernet config

        Args:
            task_type: task type.
            dataset_type: dataset type.
        
        Returns:
            param_hypernet: specified parameters for the hypernetwork.
        '''
        if task_type == TaskType.RECURSIVE.value or dataset_type == DatasetType.MNIST:
             param_hypernet: Dict[str, float] = {
                'hidden_size': 64,
                'block_num': 2,
                'structure_fourier_n_freqs': 32,
                'structure_n_heads': 4,
                'structure_n_layers': 1,
                'index_fourier_n_freqs': 1024
            }
        else:
            param_hypernet: Dict[str, float] = {
                'hidden_size': 128,
                'block_num': 2,
                'structure_fourier_n_freqs': 32,
                'structure_n_heads': 4,
                'structure_n_layers': 1,
                'index_fourier_n_freqs': 2048
            }
        # Override the parameters if ablation parameters are provided
        if ablation_index_fourier_n_freqs is not None:
            param_hypernet['index_fourier_n_freqs'] = ablation_index_fourier_n_freqs
        if ablation_block_num is not None:
            param_hypernet['block_num'] = ablation_block_num
        if ablation_hidden_size is not None:
            param_hypernet['hidden_size'] = ablation_hidden_size
        # Default index encoding type is "fourier", other options are "raw" and "positional"
        param_hypernet['index_encoding_type'] = 'fourier'
        if ablation_index_encoding_type is not None:
            param_hypernet['index_encoding_type'] = ablation_index_encoding_type
        param_hypernet['index_positional_n_freqs'] = ablation_index_positional_n_freqs
        param_hypernet['index_positional_sigma'] = ablation_index_positional_sigma
        return param_hypernet
        
    @staticmethod
    def get_basenet_config(task_type: int,
                           dataset_type: int,
                           model_type: int,
                           mix_status: str,
                           mix_type: str = 'depth_and_width',
                           cnn_layer_num_per_stage: int = 6
                           ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        '''
        Get basenet config for training

        Args:
            task_type: task type.
            dataset_type: dataset type.
            model_type: model type.
            mix_status: mixing status of the model, "single" or "mixed".
            mix_type: mixing type of the model, "depth" or "width" or "depth_and_width", only used for CIFAR10 CNN multi model.
            cnn_layer_num_per_stage: number of CNN layers per stage, only used for CIFAR10 CNN single model.

        Returns:
            param_specified: specified parameters for the model.
            param_sampled: distribution of the parameters for sampling.
        '''
        if task_type == TaskType.IMAGE_CLASSIFICATION.value:
            if dataset_type == DatasetType.MNIST.value:
                if mix_status == "single":
                    if model_type == ModelType.MLP.value:
                        return image_classification.mnist_single_mlp.get_basenet_config()
                    elif model_type == ModelType.CNN.value:
                        return image_classification.mnist_single_cnn.get_basenet_config()
                elif mix_status == "mixed":
                    if model_type == ModelType.MLP.value:
                        return image_classification.mnist_mixed_mlp.get_basenet_config()
                    elif model_type == ModelType.CNN.value:
                        return image_classification.mnist_mixed_cnn.get_basenet_config()
            elif dataset_type == DatasetType.MNIST_3D.value:
                if mix_status == "single":
                    return image_classification.mnist_3d_single_cnn.get_basenet_config()
                elif mix_status == "mixed":
                    return image_classification.mnist_3d_mixed_cnn.get_basenet_config()
            elif dataset_type == DatasetType.CIFAR10.value:
                if mix_status == "single":
                    return image_classification.cifar10_single_cnn.get_basenet_config(cnn_layer_num_per_stage)
                elif mix_status == "mixed":
                    return image_classification.cifar10_mixed_cnn.get_basenet_config(mix_type)
        elif task_type == TaskType.GRAPH_NODE_CLASSIFICATION.value:
            if mix_status == "single":
                if model_type == ModelType.GCN.value:
                    return graph_node_classification.single_gcn.get_basenet_config(dataset_type)
                elif model_type == ModelType.GAT.value:
                    return graph_node_classification.single_gat.get_basenet_config(dataset_type)
            elif mix_status == "mixed":
                if model_type == ModelType.GCN.value:
                    return graph_node_classification.mixed_gcn.get_basenet_config(dataset_type)
                elif model_type == ModelType.GAT.value:
                    return graph_node_classification.mixed_gat.get_basenet_config(dataset_type)
        elif task_type == TaskType.TEXT_CLASSIFICATION.value:
            if mix_status == "single":
                return text_classification.single_transformer.get_basenet_config(dataset_type)
            elif mix_status == "mixed":
                return text_classification.mixed_transformer.get_basenet_config(dataset_type)
        elif task_type == TaskType.FORMULA_REGRESSION.value:
            if mix_status == "single":
                return formula_regression.single_kan.get_basenet_config(dataset_type)
            elif mix_status == "mixed":
                return formula_regression.mixed_kan.get_basenet_config(dataset_type)
        else:
            # Recursive task
            return recursive.recursive.get_basenet_config(dataset_type)
        raise ValueError(f"Invalid task type {task_type} or dataset type {dataset_type}"
                         f" or model type {model_type} or mix status {mix_status}")
    
    @staticmethod
    def get_basenet_test_config(task_type: int,
                                dataset_type: int,
                                model_type: int,
                                mix_status: str,
                                mix_type: str = 'depth_and_width',
                                cnn_layer_num_per_stage: int = 6
                           ) -> List[Dict[str, float]]:
        '''
        Get basenet config for testing

        Args:
            task_type: task type.
            dataset_type: dataset type.
            model_type: model type.
            mix_status: mixing status of the model, "single" or "mixed".
            mix_type: mixing type of the model, "depth" or "width" or "depth_and_width", only used for CIFAR10 CNN multi model.
            cnn_layer_num_per_stage: number of CNN layers per stage, only used for CIFAR10 CNN single model.
            
        Returns:
            params_test: List of specified parameters for testing.
        '''
        if task_type == TaskType.IMAGE_CLASSIFICATION.value:
            if dataset_type == DatasetType.MNIST.value:
                if mix_status == "single":
                    if model_type == ModelType.MLP.value:
                        return image_classification.mnist_single_mlp.get_basenet_test_config()
                    elif model_type == ModelType.CNN.value:
                        return image_classification.mnist_single_cnn.get_basenet_test_config()
                elif mix_status == "mixed":
                    if model_type == ModelType.MLP.value:
                        return image_classification.mnist_mixed_mlp.get_basenet_test_config()
                    elif model_type == ModelType.CNN.value:
                        return image_classification.mnist_mixed_cnn.get_basenet_test_config()
            elif dataset_type == DatasetType.MNIST_3D.value:
                if mix_status == "single":
                    return image_classification.mnist_3d_single_cnn.get_basenet_test_config()
                elif mix_status == "mixed":
                    return image_classification.mnist_3d_mixed_cnn.get_basenet_test_config()
            elif dataset_type == DatasetType.CIFAR10.value:
                if mix_status == "single":
                    return image_classification.cifar10_single_cnn.get_basenet_test_config(cnn_layer_num_per_stage)
                elif mix_status == "mixed":
                    return image_classification.cifar10_mixed_cnn.get_basenet_test_config(mix_type)
        elif task_type == TaskType.GRAPH_NODE_CLASSIFICATION.value:
            if mix_status == "single":
                if model_type == ModelType.GCN.value:
                    return graph_node_classification.single_gcn.get_basenet_test_config(dataset_type)
                elif model_type == ModelType.GAT.value:
                    return graph_node_classification.single_gat.get_basenet_test_config(dataset_type)
            elif mix_status == "mixed":
                if model_type == ModelType.GCN.value:
                    return graph_node_classification.mixed_gcn.get_basenet_test_config(dataset_type)
                elif model_type == ModelType.GAT.value:
                    return graph_node_classification.mixed_gat.get_basenet_test_config(dataset_type)
        elif task_type == TaskType.TEXT_CLASSIFICATION.value:
            if mix_status == "single":
                return text_classification.single_transformer.get_basenet_test_config(dataset_type)
            elif mix_status == "mixed":
                return text_classification.mixed_transformer.get_basenet_test_config(dataset_type)
        elif task_type == TaskType.FORMULA_REGRESSION.value:
            if mix_status == "single":
                return formula_regression.single_kan.get_basenet_test_config(dataset_type)
            elif mix_status == "mixed":
                return formula_regression.mixed_kan.get_basenet_test_config(dataset_type)
        else:
            # Recursive task
            return recursive.recursive.get_basenet_test_config(dataset_type)
        raise ValueError(f"Invalid task type {task_type} or dataset type {dataset_type}"
                         f" or model type {model_type} or mix status {mix_status}")

    @staticmethod
    @jit_script
    def cal_weight_stats(layers_params: List[List[torch.Tensor]]) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        '''
        Calculate the weight statistics for the hypernetwork.

        layers_params: the parameters for each layer.

        Returns:
            mean_stds_layers: the weight statistics for each layer.
        '''
        # Calculate the statistics of the weights
        mean_stds_layers: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

        for layer_idx, layer_params in enumerate(layers_params):
            mean_stds_layers.append(cal_layer_mean_stds(layer_params))
        return mean_stds_layers

    @staticmethod
    @jit_script
    def cal_init_loss(layers_params: List[List[torch.Tensor]],
                      target_layers_stats: List[List[Tuple[float, float]]]
                      ) -> torch.Tensor:
        '''
        Calculate the initialization loss for the hypernetwork.

        layers_params: the parameters for each layer.
        target_layers_stats: the target statistics for each layer.

        Returns:
            loss: the initialization loss.
        '''
        # Calculate the statistics of the weights
        mean_stds_layers: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

        for layer_idx, layer_params in enumerate(layers_params):
            mean_stds_layers.append(cal_layer_mean_stds(layer_params))


        # Flatten the list of mean and std tensors of source and target
        mean_stds_layers_flat: List[torch.Tensor] = []
        target_layers_stats_flat: List[float] = []
        for layer_idx in range(len(mean_stds_layers)):
            for component_idx in range(len(mean_stds_layers[layer_idx])):
                for statistic_idx in range(len(mean_stds_layers[layer_idx][component_idx])):
                    if mean_stds_layers[layer_idx][component_idx][statistic_idx].numel() > 0:
                        mean_stds_layers_flat.append(mean_stds_layers[layer_idx][component_idx][statistic_idx])
                        target_layers_stats_flat.append(target_layers_stats[layer_idx][component_idx][statistic_idx])
   
        # Concatenate the mean and std tensors
        mean_stds_layers_flat: torch.Tensor = torch.stack(mean_stds_layers_flat, dim=0)
        target_layers_stats_flat: torch.Tensor = torch.tensor(
            target_layers_stats_flat, device='cuda'
        )
        loss: torch.Tensor = torch.mean((mean_stds_layers_flat - target_layers_stats_flat).pow(2))
        return loss

    def initializing(self,
                     hypernet: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler.SequentialLR,
                     steps: int,
                     writer: SummaryWriter,
                     log_freq: int = 100,
                     check_weight_stats: bool = False
                     ) -> None:
        '''
        Initializing the weights generated by the hypernetwork.

        Args:
            - hypernet: hypernetwork.
            - optimizer: optimizer.
            - scheduler: scheduler.
            - steps: number of steps.
            - writer: tensorboard writer.
            - log_freq: log frequency.
            - check_weight_stats: whether to check the weight statistics during initialization.
        '''
        # Get gradient scaler
        scaler: Optional[GradScaler] = get_grad_scaler()
        hypernet.train()

        # Reset model set loaders if available
        self.task_params.reset_model_set_data_loaders()

        # Initialize the running statistics
        running_loss: torch.Tensor = torch.tensor(0.0, device='cuda')
        running_step: int = 0
        for step in tqdm(range(steps), desc='Initializing'):
            # Sample global and local modes
            global_mode, local_mode = next(self.task_params)

            optimizer.zero_grad()
            with get_precision_ctx():
                # Forward pass for initialization
                weights, layers_params, layers_param_shapes, global_mode, local_mode = \
                    Task.model_params_generate(
                        hypernet=hypernet,
                        task_params=self.task_params,
                        test_config_idx=None,
                        preallocated_memory=self.preallocated_memory,
                        global_mode=global_mode,
                        local_mode=local_mode
                        )
                # Calculate the target initialization statistics
                target_layers_stats: List[List[Tuple[float, float]]] = ModelUtils.get_weights_initial_statistic(
                    local_mode=local_mode
                )
                # Iterate the weights and calculate the distribution matching loss
                loss = Task.cal_init_loss(
                    layers_params=layers_params,
                    target_layers_stats=target_layers_stats
                )

                if check_weight_stats:
                    # Check the weight statistics, also log loss per step
                    writer.add_scalar(f"Initialize/loss_step", loss.item(), step)
                    # Log the weight statistics
                    mean_stds_layers = Task.cal_weight_stats(layers_params)
                    for layer_idx in range(len(mean_stds_layers)):
                        for component_idx in range(len(mean_stds_layers[layer_idx])):
                            for statistic_idx in range(len(mean_stds_layers[layer_idx][component_idx])):
                                if mean_stds_layers[layer_idx][component_idx][statistic_idx].numel() > 0:
                                    if statistic_idx == 0:
                                        stat_str = "mean"
                                    else:
                                        stat_str = "std"
                                    writer.add_scalar(f"Initialize_weight_stats/layer_{layer_idx}/"
                                        f"component_{component_idx}/{stat_str}", mean_stds_layers[layer_idx][component_idx][statistic_idx].item(), step)

                # Update running statistics
                running_loss += loss.detach()
                running_step += 1
                # Backward pass
                if scaler is not None:
                    # Use mixed precision
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Full precisioin
                    loss.backward()
                    optimizer.step()
                # Log the training statistic
                if (step + 1) % log_freq == 0 or (step + 1) == steps:
                    avg_loss = (running_loss / running_step).item()
                    writer.add_scalar("Initialize/loss", avg_loss, step)
                    # Log lr
                    writer.add_scalar("Initialize/lr", optimizer.param_groups[0]['lr'], step)
                    print(f"Step {step + 1}/{steps} [Initializing]: Loss: {avg_loss}")
                    # Reset the running statistics
                    running_loss.fill_(0.0)
                    running_step = 0
                # Update the learning rate
                scheduler.step()

    def training(self,
                 hypernet: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.SequentialLR,
                 epochs: int,
                 writer: SummaryWriter,
                 validate_on_test_param: bool = False,
                 check_freq: int = 1,
                 check_weight_stats: bool = False,
                 test: bool = False,
                 grad_clip: bool = False
                 ) -> List[Dict[str, Dict[str, float]]]:
        '''
        Training the hypernetwork.

        Args:
            - hypernet: hypernetwork.
            - optimizer: optimizer.
            - scheduler: scheduler.
            - epochs: number of epochs.
            - writer: tensorboard writer.
            - validate_on_test_param: whether to validate on test parameters.
            - check_freq: check frequency.
            - check_weight_stats: whether to check the weight statistics during training.
            - test: whether to test the model on the test set after training.
            - grad_clip: whether to apply gradient clipping during training.
        '''
        # Get gradient scaler
        scaler: Optional[GradScaler] = get_grad_scaler()
        task_type: int = int(self.task_params.param_specified["task_type"])

        # Reset model set loaders if available
        self.task_params.reset_model_set_data_loaders()

        # Evaluate results
        eval_results: List[Dict[str, Dict[str, float]]] = []

        step: int = 0
        for epoch in range(epochs):
            # Training mode
            hypernet.train()
            # Initialize the running statistics
            running_loss = torch.tensor(0.0, device='cuda')
            total = 0
            running_corrects = torch.tensor(0, device='cuda')
            total_classify = 0
            # Iterate over the training data
            if task_type != TaskType.GRAPH_NODE_CLASSIFICATION.value:
                # Non-graph data training
                for inputs, targets in tqdm(self.task_params.train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Training]'):
                    inputs, targets = inputs.cuda(), targets.cuda()

                    # Sample global and local modes
                    global_mode, local_mode = next(self.task_params)

                    optimizer.zero_grad()
                    with get_precision_ctx():
                        # Forward pass
                        if task_type == TaskType.TEXT_CLASSIFICATION.value:
                            # Get padding mask for text classification
                            padding_mask = (inputs == PAD_TOKEN_IDX)
                            x = [inputs, padding_mask]
                        else:
                            x = [inputs]
                        outputs, _, _, layers_params = Task.model_infer(
                            x=x,
                            hypernet=hypernet,
                            task_params=self.task_params,
                            test_config_idx=None,
                            preallocated_memory=self.preallocated_memory,
                            training=True,
                            global_mode=global_mode,
                            local_mode=local_mode
                            )
                        # Compute the loss
                        loss = self.task_params.loss_func(outputs, targets)
                        if check_weight_stats:
                            # Log the training statistics every step
                            writer.add_scalar('Loss_step/Train', loss.item(), step)                             
                            # Log the weight statistics
                            mean_stds_layers = Task.cal_weight_stats(layers_params)
                            for layer_idx in range(len(mean_stds_layers)):
                                for component_idx in range(len(mean_stds_layers[layer_idx])):
                                    for statistic_idx in range(len(mean_stds_layers[layer_idx][component_idx])):
                                        if mean_stds_layers[layer_idx][component_idx][statistic_idx].numel() > 0:
                                            if statistic_idx == 0:
                                                stat_str = "mean"
                                            else:
                                                stat_str = "std"
                                            writer.add_scalar(f"Training_weight_stats/layer_{layer_idx}/"
                                                f"component_{component_idx}/{stat_str}", mean_stds_layers[layer_idx][component_idx][statistic_idx].item(), step)
                            # Log the outputs statistics
                            writer.add_scalar('Training_outputs/mean', outputs.mean().item(), step)
                            writer.add_scalar('Training_outputs/std', outputs.std(correction=0).item(), step)

                        # Update the running statistics
                        data_size: int = targets.size(0)
                        running_loss += loss.detach() * data_size
                        total += data_size
                        if task_type in [TaskType.IMAGE_CLASSIFICATION.value,
                                        TaskType.TEXT_CLASSIFICATION.value]:
                            # Calculate the accuracy
                            _, preds = torch.max(outputs, 1)
                            corrects = torch.sum(preds == targets)
                            running_corrects += corrects
                            total_classify += data_size
                    # Backward pass
                    if scaler is not None:
                        # Use mixed precision
                        scaler.scale(loss).backward()

                        if grad_clip:
                            # Apply gradient clipping
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)

                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Use full precision
                        loss.backward()

                        if grad_clip:
                            # Apply gradient clipping
                            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)

                        optimizer.step()
                    step += 1
            else:
                # Graph data training
                data: BaseData = self.task_params.graph_dataset[0].cuda()
                x, edge_index = data.x, data.edge_index

                # Sample global and local modes
                global_mode, local_mode = next(self.task_params)

                optimizer.zero_grad()
                with get_precision_ctx():
                    # Forward pass
                    outputs, _, _, layers_params = Task.model_infer(
                        x=[x, edge_index],
                        hypernet=hypernet,
                        task_params=self.task_params,
                        test_config_idx=None,
                        preallocated_memory=self.preallocated_memory,
                        training=True,
                        global_mode=global_mode,
                        local_mode=local_mode
                        )
                    # Compute the loss
                    loss = self.task_params.loss_func(outputs[data.train_mask], data.y[data.train_mask])
                    if check_weight_stats:
                        # Log the training statistics every step
                        writer.add_scalar('Loss_step/Train', loss.item(), step)                         
                        # Log the weight statistics
                        mean_stds_layers = Task.cal_weight_stats(layers_params)
                        for layer_idx in range(len(mean_stds_layers)):
                            for component_idx in range(len(mean_stds_layers[layer_idx])):
                                for statistic_idx in range(len(mean_stds_layers[layer_idx][component_idx])):
                                    if mean_stds_layers[layer_idx][component_idx][statistic_idx].numel() > 0:
                                        if statistic_idx == 0:
                                            stat_str = "mean"
                                        else:
                                            stat_str = "std"
                                        writer.add_scalar(f"Training_weight_stats/layer_{layer_idx}/"
                                            f"component_{component_idx}/{stat_str}", mean_stds_layers[layer_idx][component_idx][statistic_idx].item(), step)
                        # Log the outputs statistics
                        writer.add_scalar('Training_outputs/mean', outputs[data.train_mask].mean().item(), step)
                        writer.add_scalar('Training_outputs/std', outputs[data.train_mask].std(correction=0).item(), step)                                            
                    # Calculate the accuracy
                    _, preds = torch.max(outputs[data.train_mask], 1)
                    corrects = torch.sum(preds == data.y[data.train_mask])
                    # Update the running statistics
                    data_size: int = data.y[data.train_mask].size(0)
                    running_loss += loss.detach() * data_size
                    running_corrects += corrects
                    total += data_size
                    total_classify += data_size
                # Backward pass
                if scaler is not None:
                    # Use mixed precision
                    scaler.scale(loss).backward()

                    if grad_clip:
                        # Apply gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Use full precision
                    loss.backward()

                    if grad_clip:
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)

                    optimizer.step()
                step += 1
            # Log the learning rate
            for group_idx, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Learning Rate/Train_{group_idx}', param_group['lr'], epoch)
                # Also log weight_decay
                writer.add_scalar(f'Weight Decay/Train_{group_idx}', param_group['weight_decay'], epoch)

            # Update the learning rate
            scheduler.step()
            # Log the training statistics
            avg_loss = (running_loss / total).item()
            writer.add_scalar('Loss/Train', avg_loss, epoch)
            print_str = f'Epoch {epoch + 1}/{epochs} [Training]: Loss: {avg_loss}'
            if total_classify > 0:
                avg_acc = (running_corrects / total_classify).item()
                writer.add_scalar('Accuracy/Train', avg_acc, epoch)
                print_str += f', Accuracy: {avg_acc}'
            print(print_str)

            # Validation, except for the case that multi-task is enabled and multi_model_mode is 'fixed'
            if not (self.task_params.model_set_params is not None and \
                    self.task_params.model_set_params.multi_model_mode == "fixed"):
                if (epoch + 1) % check_freq == 0 or (epoch + 1) == epochs:
                    eval_result: Dict[str, Dict[str, float]] = {}
                    # Validate on training set
                    train_result: Dict[str, float] = self.testing(hypernet=hypernet,
                        epoch=epoch,
                        epochs=epochs,
                        writer=writer,
                        on_test_param=False,
                        test_split=DatasetSplit.TRAIN)[0]
                    eval_result[DatasetSplit.TRAIN.name] = train_result
                    if validate_on_test_param:
                        self.testing(hypernet=hypernet,
                                    epoch=epoch,
                                    epochs=epochs,
                                    writer=writer,
                                    on_test_param=True,
                                    test_split=DatasetSplit.TRAIN)                                    
                    if self.task_params.validate:
                        # Validate on train parameters
                        val_result: Dict[str, float] = self.testing(hypernet=hypernet,
                            epoch=epoch,
                            epochs=epochs,
                            writer=writer,
                            on_test_param=False,
                            test_split=DatasetSplit.VAL)[0]
                        eval_result[DatasetSplit.VAL.name] = val_result
                        # Validate on test parameters
                        if validate_on_test_param:               
                            # Validate on validation set
                            self.testing(hypernet=hypernet,
                                        epoch=epoch,
                                        epochs=epochs,
                                        writer=writer,
                                        on_test_param=True,
                                        test_split=DatasetSplit.VAL)
                    if self.task_params.validate or test:
                        # Validate on test parameters for the convenience of observation only,
                        #   DO NOT use the test set for hyperparameter tuning
                        test_result: Dict[str, float] = self.testing(hypernet=hypernet,
                            epoch=epoch,
                            epochs=epochs,
                            writer=writer,
                            on_test_param=False,
                            test_split=DatasetSplit.TEST)[0]
                        eval_result[DatasetSplit.TEST.name] = test_result
                        # Validate on test parameters
                        if validate_on_test_param:           
                            # Validate on test set, for the convenience of observation only,
                            #   DO NOT use the test set for hyperparameter tuning
                            self.testing(hypernet=hypernet,
                                        epoch=epoch,
                                        epochs=epochs,
                                        writer=writer,
                                        on_test_param=True,
                                        test_split=DatasetSplit.TEST)
                    # Update eval result list    
                    eval_results.append(eval_result)
        return eval_results

    def testing(self,
                hypernet: torch.nn.Module,
                epoch: int,
                epochs: int,
                writer: SummaryWriter,
                on_test_param: bool = False,
                test_split: DatasetSplit = DatasetSplit.TEST,
                ) -> List[Dict[str, float]]:
        '''
        Testing

        Args:
            - hypernet: hypernetwork.
            - epoch: current epoch.
            - epochs: total number of epochs.
            - writer: tensorboard writer.
            - on_test_param: whether to test the data on test parameters.
            - test_split: test_split, train, val or test.
        '''
        if on_test_param:
            # Use the test parameters for testing on test data
            if self.task_params.test_configs is None:
                raise ValueError("Test configs are not initialized")
            test_configs_idxes = [idx for idx in range(len(self.task_params.test_configs))]
        else:
            # Use the training parameters for testing on test data
            test_configs_idxes = [None]
        # Iterate over the test parameters
        results: List[Dict[str, float]] = []
        for test_config_idx in test_configs_idxes:
            log_str = f'data_{test_split.name}_param_{"train" if test_config_idx is None else f"test_{test_config_idx}"}'
            loss, acc, rmse = Task.model_test(
                hypernet=hypernet,
                task_params=self.task_params,
                test_config_idx=test_config_idx,
                preallocated_memory=self.preallocated_memory,
                epoch=epoch,
                epochs=epochs,
                writer=writer,
                log_str=log_str,
                test_split=test_split,
                global_mode=None,
                local_mode=None
            )
            result: Dict[str, float] = {'loss': loss}
            if acc is not None:
                result['accuracy'] = acc
            if rmse is not None:
                result['rmse'] = rmse
            results.append(result)
        return results

    def test_model_set(self,
                       hypernet: torch.nn.Module,
                       epoch: int,
                       epochs: int,
                       writer: SummaryWriter,
                       test_data_split: DatasetSplit = DatasetSplit.TEST,
                       test_model_split: Optional[DatasetSplit] = None
                    ) -> Dict[str, float]:
        '''
        Test a specific model set on a specific data set, via testing all the models 
            in the model set on the specified data set one by one.
        
        Args:
            - hypernet: hypernetwork.
            - epoch: current epoch.
            - epochs: total number of epochs.
            - writer: tensorboard writer.
            - test_data_split: the split of the data set to be tested, train, val or test.
            - test_model_split: the split of the model set to be tested, train, val or test.
        '''
        model_set_params = self.task_params.model_set_params
        if model_set_params is None or \
                model_set_params.multi_model_mode != "fixed":
            raise ValueError(
                "The model set is not enabled or the multi_model_mode is not 'fixed'."
            )
        
        # Retrieve the corresponding model set data loader
        if model_set_params.test_mode == "full_set":
            model_loader = model_set_params.loader_S_prime
        elif model_set_params.test_mode == "hold_out":
            if test_model_split is None:
                raise ValueError("test_model_split must be specified when test_mode is 'hold_out'")
            if test_model_split == DatasetSplit.TRAIN:
                model_loader = model_set_params.loader_S_train_prime
            elif test_model_split == DatasetSplit.VAL:
                model_loader = model_set_params.loader_S_val_prime
            else:
                model_loader = model_set_params.loader_S_test_prime
        else:
            raise ValueError(f"Invalid model set test mode {model_set_params.test_mode}")

        # Traverse the model set data loader
        model_modes: List[Tuple[Dict[str, float], List[Dict[str, float]]]] = []
        losses: List[float] = []
        accs: List[Optional[float]] = []
        rmses: List[Optional[float]] = []
        for model_idx, (global_mode, local_mode) in tqdm(enumerate(model_loader), desc=f'Model Set Testing'):
            log_str = f"model_{model_set_params.test_mode}"
            if model_set_params.test_mode == "hold_out":
                log_str += f"_{test_model_split.name}"
            log_str += f"_idx_{model_idx}"
            log_str += f"_data_{test_data_split.name}"
            loss, acc, rmse = Task.model_test(
                hypernet=hypernet,
                task_params=self.task_params,
                test_config_idx=None,  
                preallocated_memory=self.preallocated_memory,
                epoch=epoch,
                epochs=epochs,
                writer=writer,
                log_str=log_str,
                test_split=test_data_split,
                global_mode=global_mode,
                local_mode=local_mode
            )
            # Save the model modes
            model_modes.append((global_mode, local_mode))
            losses.append(loss)
            accs.append(acc)
            rmses.append(rmse)
        
        # Calculate the average of the losses and accuracies, etc.
        result: Dict[str, float] = {}        
        avg_loss = sum(losses) / len(losses)
        result['loss'] = avg_loss
        if accs[0] is not None:
            avg_acc = sum(accs) / len(accs)
            result['accuracy'] = avg_acc
        else:
            avg_acc = None
        if rmses[0] is not None:
            avg_rmse = sum(rmses) / len(rmses)
            result['rmse'] = avg_rmse
        else:
            avg_rmse = None
        # Log the results
        log_str = f"model_{model_set_params.test_mode}"
        if model_set_params.test_mode == "hold_out":
            log_str += f"_{test_model_split.name}"
        log_str += f"_average"
        log_str += f"_data_{test_data_split.name}"
        # Log to tensorboard
        writer.add_scalar(f'Loss_ModelSet/{log_str}', avg_loss, epoch)
        print_str = f'Epoch {epoch + 1}/{epochs} Model Set Testing [{log_str}]: Loss: {avg_loss}'
        if avg_acc is not None:
            writer.add_scalar(f'Accuracy_ModelSet/{log_str}', avg_acc, epoch)
            print_str += f', Accuracy: {avg_acc}' 
        if avg_rmse is not None:
            writer.add_scalar(f'RMSE_ModelSet/{log_str}', avg_rmse, epoch)
            print_str += f', RMSE: {avg_rmse}'
        print(print_str)
        return result

    def exporting(self,
                  hypernet: torch.nn.Module,
                  sample_num: int = 1,
                  export_pth: bool = False,
                  export_description_file: bool = False,
                  save_dir: str = 'outputs',
                  on_test_param: bool = False
                  ) -> Tuple[List[torch.nn.Module],
                             List[Tuple[Dict[str, float], List[Dict[str, float]]]]]:
        '''
        Model exporting

        Args:
            - hypernet: hypernetwork.
            - sample_num: number of networks to be sampled for exporting
            - export_pth: whether to export to pth.
            - export_description_file: whether to export the description file.
            - save_dir: directory to save the exported models.
            - on_test_param: whether to export the model on test parameters.

        Returns:
            - networks: list of exported networks.
            - networks_description: list of exported networks' descriptions.
        '''
        if on_test_param:
            # Use the test parameters for testing on test data
            if self.task_params.test_configs is None:
                raise ValueError("Test configs are not initialized")
            test_configs_idxes = [idx for idx in range(len(self.task_params.test_configs))]
        else:
            # Use the training parameters for testing on test data
            test_configs_idxes = [None] * sample_num
            # Reset model set loaders if available
            self.task_params.reset_model_set_data_loaders()            
        # Networks
        networks: List[torch.nn.Module] = []
        # Network descriptions, i.e., global and local structures
        networks_description: List[Tuple[Dict[str, float], List[Dict[str, float]]]] = []            
        # Iterate over the configs
        for sample_idx, test_config_idx in enumerate(test_configs_idxes):
            hypernet.eval()
            # Sample global and local modes
            if not on_test_param:
                global_mode, local_mode = next(self.task_params)
            else:
                global_mode, local_mode = None, None
            with torch.no_grad():
                with get_precision_ctx():
                    # Generate the network
                    network, global_mode, local_mode = Task.model_modularize(
                        hypernet=hypernet,
                        task_params=self.task_params,
                        test_config_idx=test_config_idx,
                        preallocated_memory=self.preallocated_memory,
                        global_mode=global_mode,
                        local_mode=local_mode
                        )
                    # Save the network
                    networks.append(network)
                    # Save the network description
                    networks_description.append((global_mode, local_mode))
                    # Save the network to file
                    if export_pth or export_description_file:
                        # Ensure the directory exists
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        # Generate name for this network
                        network_name = f'{save_dir}/network_{"test_" if on_test_param else ""}{sample_idx}'
                        # Save the network as pth
                        if export_pth:
                            pth_name = f'{network_name}.pth'
                            torch.save(network, pth_name)
                            print(f"Network saved as [pth] {pth_name}")
                        # Save the description file
                        if export_description_file:
                            description_name = f'{network_name}.json'
                            # Make sure that the modes is with float values
                            global_mode = {k: float(v) for k, v in global_mode.items()}
                            local_mode = [{k: float(v) for k, v in local_mode_layer.items()} for local_mode_layer in local_mode]
                            with open(description_name, 'w') as f:
                                json.dump({'global structure': global_mode,
                                        'local structure': local_mode}, 
                                        f, indent=4)
                            print(f"Network description saved as [json] {description_name}")
        return networks, networks_description
