'''
Solving recursive tasks
'''
from .multi_task import *
from model.model.recursive_model import RecursiveModelUtils
from model.hyper_network import HyperNetwork, FourierEncoder
from typing import Callable

class HypernetProxy:
    '''
    Proxy for hypernet, an equivalent representation of the model.
    '''
    def __init__(self,
                 global_mode: Dict[str, float],
                 local_mode: List[Dict[str, float]],
                 layers_params: List[List[torch.Tensor]],
                 layers_param_shapes: List[List[List[int]]],
                 structure_fourier_encoder: FourierEncoder,
                 index_fourier_encoder: FourierEncoder,
                 training: bool
                 ) -> None:
        '''
        Initialize model proxy
        Args:
            global_mode: Global structure of the model.
            local_mode: Local structures of the model.
            layers_params: Parameters for each layer.
            layers_param_shapes: Shapes of the parameters for each layer.
            global_fourier_encoder: Fourier encoder for global structure.
            local_fourier_encoder: Fourier encoder for local structures.
            index_fourier_encoder: Fourier encoder for index encoding.
            training: Training mode.
        '''
        self.global_mode = global_mode
        self.local_mode = local_mode
        self.layers_params = layers_params
        self.layers_param_shapes = layers_param_shapes
        # Notice that the encoder contains both input preprocessing 
        #   and fourier encoding
        self.structure_fourier_encoder = structure_fourier_encoder
        self.index_fourier_encoder = index_fourier_encoder
        self.training = training

class RecursiveTask(MultiTask):
    '''
    Solving recursive tasks
    '''
    def __init__(self,
                 parent_tasks_params: List[TaskParams],
                 child_tasks: List[Optional[MultiTask]],
                 parent_tasks_prob: Optional[List[float]] = None,
                 ) -> None:
        '''
        Initialize recursive tasks
        Args:
            parent_tasks_params (List[TaskParams]): List of parent task params
            child_tasks (List[MultiTask]): List of child tasks
            parent_tasks_prob (Optional[List[float]]): List of parent task probabilities
        '''
        # Initialize parent task
        super().__init__(parent_tasks_params, parent_tasks_prob)
        # Link parent task and child tasks
        self.child_tasks: List[Optional[MultiTask]] = child_tasks

    @staticmethod
    def calculate_max_memory_size(tasks_params: TaskParams,
                                  child_tasks: List[MultiTask] 
                                  ) -> Tuple[int, int, int, int, int]:
        '''
        Calculate the maximum memory size for structure and index encoding for 
            each parent task and child task.

        Returns:
            global_structure_size: the maximum memory size for global structure.
            local_structure_size: the maximum memory size for local structures.
            encode_memory_size: the maximum memory size for encoding.
            shared_element_buffer_size: the maximum memory size for shared elements.
            arange_tensor_size: the maximum memory size for arange tensor.          
        '''
        # Calculate the size of the preallocated memory for each parent task first
        global_structure_size_max, local_structure_size_max, encode_memory_size_max, \
            shared_element_buffer_size_max, arange_tensor_size_max = \
                MultiTask.calculate_max_memory_size(tasks_params)   
        # Iterate through each child task and calculate the max memory size
        for child_task in child_tasks:
            if child_task is None:
                # Skip the None child tasks
                continue
            elif isinstance(child_task, RecursiveTask):
                # Non-leaf child tasks
                global_structure_size, local_structure_size, encode_memory_size, \
                    shared_element_buffer_size, arange_tensor_size = \
                    RecursiveTask.calculate_max_memory_size(
                        child_task.tasks_params, child_task.child_tasks)
            else:
                # Leaf child tasks
                global_structure_size, local_structure_size, encode_memory_size, \
                    shared_element_buffer_size, arange_tensor_size = \
                    MultiTask.calculate_max_memory_size(child_task.tasks_params)
            # Update the max size for each task
            global_structure_size_max = max(global_structure_size_max, global_structure_size)
            local_structure_size_max = max(local_structure_size_max, local_structure_size)
            encode_memory_size_max = max(encode_memory_size_max, encode_memory_size)
            shared_element_buffer_size_max = max(shared_element_buffer_size_max, shared_element_buffer_size)
            arange_tensor_size_max = max(arange_tensor_size_max, arange_tensor_size)

        return global_structure_size_max, local_structure_size_max, encode_memory_size_max, \
            shared_element_buffer_size_max, arange_tensor_size_max
    
    @staticmethod
    def assign_preallocated_memory_to_child_tasks(
            preallocated_memory: PreallocatedMemory,
            child_tasks: List[MultiTask]
        ) -> None:
        '''
        Assign preallocated memory to each child task
        '''
        for child_task in child_tasks:
            if child_task is None:
                # Skip the None child tasks
                continue
            child_task._preallocated_memory = preallocated_memory
            if isinstance(child_task, RecursiveTask):
                # Non-leaf child tasks
                RecursiveTask.assign_preallocated_memory_to_child_tasks(
                    preallocated_memory, child_task.child_tasks)
    
    @property
    def preallocated_memory(self) -> PreallocatedMemory:
        '''
        Preallocated memory for each task
        '''
        if self._preallocated_memory is None:
            # Calculate the size of the preallocated memory for each parent task first
            global_structure_size_max, local_structure_size_max, encode_memory_size_max, \
                shared_element_buffer_size_max, arange_tensor_size_max = \
                    RecursiveTask.calculate_max_memory_size(
                        tasks_params=self.tasks_params, 
                        child_tasks=self.child_tasks)
            # Preallocate memory for each task
            self._preallocated_memory = PreallocatedMemory(
                global_structure_size=global_structure_size_max,
                local_structure_size=local_structure_size_max,
                encode_memory_size=encode_memory_size_max,
                shared_element_buffer_size=shared_element_buffer_size_max,
                arange_tensor_size=arange_tensor_size_max
            )
            # Assign preallocated memory to each child task
            RecursiveTask.assign_preallocated_memory_to_child_tasks(
                self._preallocated_memory, self.child_tasks)
        return self._preallocated_memory

    @staticmethod
    def retrieve_encode_input_minmax(tasks_params: TaskParams,
                                     child_tasks: List[MultiTask]
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
        global_structure_minmax, local_structure_minmax, index_encoding_minmax = \
            MultiTask.retrieve_encode_input_minmax(tasks_params=tasks_params)
        # Iterate through each child task and calculate the min and max values
        for child_task in child_tasks:
            if child_task is None:
                # Skip the None child tasks
                continue
            elif isinstance(child_task, RecursiveTask):
                # Non-leaf child tasks
                global_structure_minmax_task, local_structure_minmax_task, index_encoding_minmax_task = \
                    RecursiveTask.retrieve_encode_input_minmax(
                        tasks_params=child_task.tasks_params, 
                        child_tasks=child_task.child_tasks)
            else:
                # Leaf child tasks
                global_structure_minmax_task, local_structure_minmax_task, index_encoding_minmax_task = \
                    MultiTask.retrieve_encode_input_minmax(
                        tasks_params=child_task.tasks_params)
            # Update the min and max values for each task
            global_structure_minmax[0] = torch.minimum(global_structure_minmax[0], global_structure_minmax_task[0])
            global_structure_minmax[1] = torch.maximum(global_structure_minmax[1], global_structure_minmax_task[1])
            local_structure_minmax[0] = torch.minimum(local_structure_minmax[0], local_structure_minmax_task[0])
            local_structure_minmax[1] = torch.maximum(local_structure_minmax[1], local_structure_minmax_task[1])
            index_encoding_minmax[0] = torch.minimum(index_encoding_minmax[0], index_encoding_minmax_task[0])
            index_encoding_minmax[1] = torch.maximum(index_encoding_minmax[1], index_encoding_minmax_task[1])

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
        return RecursiveTask.retrieve_encode_input_minmax(
            tasks_params=self.tasks_params,
            child_tasks=self.child_tasks)

    @staticmethod
    def model_params_generate(hypernet_proxy: HypernetProxy,
                              task_params: TaskParams,
                              test_config_idx: Optional[int],
                              preallocated_memory: PreallocatedMemory,
                              ) -> Tuple[torch.Tensor,
                                         List[List[torch.Tensor]], List[List[List[int]]], 
                                         Dict[str, float], List[Dict[str, float]]]:
        '''
        Generate model parameters through proxy hypernetworks.

        Args:
            hypernet_proxy: hypernet proxy, an equivalent representation of the model.
            task_params: task parameters.
            test_config_idx: index of the test configuration. None if use the training configuration.
            preallocated_memory: preallocated memory for structure and index encoding.

        Returns:
            weights: the vanilla weights generated by the hypernetwork.
            layers_params: the parameters for each layer.
            layers_param_shapes: the shapes of the parameters for each layer.
            global_mode: the global structure of the model.
            local_mode: the local structures of the model.
        '''
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
        global_mode: Dict[str, float]
        local_mode: List[Dict[str, float]]
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
        # Preprocess, i.e., the input normalization & fourier encoding
        structure: torch.Tensor = torch.cat((global_structure.expand(1, local_structures.size(1), -1), 
                            local_structures), dim=2)
        structure_fourier: torch.Tensor = \
            hypernet_proxy.structure_fourier_encoder(structure)
        index_fourier: torch.Tensor = \
            hypernet_proxy.index_fourier_encoder(idxes)
        # Pass the model encoding to the hypernetwork to generate the weights        
        weights: torch.Tensor = RecursiveModelUtils.apply_weights(
            structure_fourier=structure_fourier,
            index_fourier=index_fourier,
            local_mode=hypernet_proxy.local_mode,
            layers_params=hypernet_proxy.layers_params,
            layers_param_shapes=hypernet_proxy.layers_param_shapes,
            training=hypernet_proxy.training
            )

        # Retrieve the params for each layer
        layers_params: List[List[torch.Tensor]] = ModelUtils.retrieve_weights(weights,
                                                                         layers_lens)
        return weights, layers_params, layers_param_shapes, global_mode, local_mode

    @staticmethod
    def model_params_generate_trace(hypernet: HyperNetwork,
                                    task: MultiTask,
                                    task_trace: List[int],
                                    test_configs_idxes: List[Optional[int]],
                                    preallocated_memory: PreallocatedMemory,
                                    training: bool
                                    ) -> List[Tuple[torch.Tensor,
                                            List[List[torch.Tensor]], List[List[List[int]]], 
                                            Dict[str, float], List[Dict[str, float]]]]:
        '''
        Generate a chain of network along the trace through hypernetworks. Nitice that
            the current task should be a recursive task.
        Example 1:
        Hypernet ---> kth RecursiveTask (task_params)
                 ---> mth child RecursiveTask (child_task.task_params[m])
                 ---> nth grand RecursiveTask (child_tasks[m].child_task.task_params[n]) 
                 ---> ith leaf Task (child_tasks[m].child_tasks[n].child_task.task_params[i])
            the child_task_trace = [k, m, n, i], child_task, child_tasks[m].child_task are
            RecursiveTask, and child_tasks[m].child_tasks[n].child_task is a MultiTask.
        Example 2:
        Hypernet ---> kth RecursiveTask (task_params)
                 ---> ith leaf Task (child_task.task_params[i])
            the child_task_trace = [k, i], child_task is a MultiTask.

        Args:
            hypernet: hypernetwork.
            task: the parent task (RecursiveTask) for the model.
            task_trace: the trace (selected indexes) of child tasks. 
            test_configs_idxes: the test configuration index for each child task.
            preallocated_memory: preallocated memory for the model.
            training: training mode.
        
        Returns:
            model_params_chains: a list of generated parameters along the trace,
        '''
        # Generated parameters by the hypernetwork
        model_params_chains: List[Tuple[torch.Tensor,
                            List[List[torch.Tensor]], List[List[List[int]]],
                            Dict[str, float], List[Dict[str, float]]]] = []
        # Calculate the weights for generated hypernetwork for child tasks
        weights, layers_params, layers_param_shapes, global_mode, local_mode = \
            Task.model_params_generate(
                hypernet=hypernet,
                task_params=task.tasks_params[task_trace[0]],
                test_config_idx=test_configs_idxes[0],
                preallocated_memory=preallocated_memory
                )
        # Append the generated parameters for the first task
        model_params_chains.append(
            (weights, layers_params, layers_param_shapes, global_mode, local_mode))
        # Generate network for child tasks
        child_task: Optional[MultiTask] = task.child_tasks[task_trace[0]]
        child_task_trace: List[int] = task_trace[1:]
        child_test_configs_idxes: List[Optional[int]] = test_configs_idxes[1:]        
        for child_task_depth in range(len(child_task_trace)):
            child_task_idx: int = child_task_trace[child_task_depth]
            child_task_params: TaskParams = child_task.tasks_params[child_task_idx]
            weights, layers_params, layers_param_shapes, global_mode, local_mode = \
                RecursiveTask.model_params_generate(
                    hypernet_proxy=HypernetProxy(
                        global_mode=global_mode,
                        local_mode=local_mode,
                        layers_params=layers_params,
                        layers_param_shapes=layers_param_shapes,
                        structure_fourier_encoder=hypernet.structure_input_network[0],
                        index_fourier_encoder=hypernet.index_input_network[0],
                        training=training),
                    task_params=child_task_params,
                    test_config_idx=child_test_configs_idxes[child_task_depth],
                    preallocated_memory=preallocated_memory
                    )
            # Append the generated parameters for the child task
            model_params_chains.append(
                (weights, layers_params, layers_param_shapes, global_mode, local_mode))
            if child_task_depth < len(child_task_trace) - 1:
                # Non-leaf child tasks, update the child tasks for next generation
                child_task = child_task.child_tasks[child_task_idx]
        return model_params_chains
    
    @staticmethod
    def model_infer(x: List[torch.Tensor],
                    hypernet: HyperNetwork,
                    task: MultiTask,
                    task_trace: List[int],
                    test_configs_idxes: List[Optional[int]],
                    preallocated_memory: PreallocatedMemory,
                    training: bool
                    ) -> Tuple[torch.Tensor,
                            Dict[str, float], List[Dict[str, float]],
                            List[Tuple[torch.Tensor,
                                List[List[torch.Tensor]], List[List[List[int]]],
                                Dict[str, float], List[Dict[str, float]]]]]:
        '''
        Whole process of model inference through hypernetworks. Nitice that
            the current task should be a recursive task.
        Example 1:
        Hypernet ---> kth RecursiveTask (task_params)
                 ---> mth child RecursiveTask (child_task.task_params[m])
                 ---> nth grand RecursiveTask (child_tasks[m].child_task.task_params[n]) 
                 ---> ith leaf Task (child_tasks[m].child_tasks[n].child_task.task_params[i])
            the child_task_trace = [k, m, n, i], child_task, child_tasks[m].child_task are
            RecursiveTask, and child_tasks[m].child_tasks[n].child_task is a MultiTask.
        Example 2:
        Hypernet ---> kth RecursiveTask (task_params)
                 ---> ith leaf Task (child_task.task_params[i])
            the child_task_trace = [k, i], child_task is a MultiTask.

        Args:
            x: a list of input tensors.
            hypernet: hypernetwork.
            task: the parent task (RecursiveTask) for the model.
            task_trace: the trace (selected indexes) of child tasks. 
            test_configs_idxes: the test configuration index for each child task.
            preallocated_memory: preallocated memory for the model.
            training: training mode.
        
        Returns:
            x: the output tensor.
            global_mode: the global structure of the model.
            local_mode: the local structures of the model.
            model_params_chains: a list of generated parameters along the trace,
        '''
        # Generated parameters by the hypernetwork
        model_params_chains: List[Tuple[torch.Tensor,
                            List[List[torch.Tensor]], List[List[List[int]]],
                            Dict[str, float], List[Dict[str, float]]]] = []        
        # Calculate the weights for generated hypernetwork for child tasks
        weights, layers_params, layers_param_shapes, global_mode, local_mode = \
            Task.model_params_generate(
                hypernet=hypernet,
                task_params=task.tasks_params[task_trace[0]],
                test_config_idx=test_configs_idxes[0],
                preallocated_memory=preallocated_memory
                )
        # Append the generated parameters for the first task
        model_params_chains.append(
            (weights, layers_params, layers_param_shapes, global_mode, local_mode))        
        child_task: Optional[MultiTask] = task.child_tasks[task_trace[0]]
        if child_task is None:
            # No child tasks
            if test_configs_idxes[0] is None:
                # Use the training configuration
                param_specified: Dict[str, float] = task.tasks_params[task_trace[0]].param_specified
            else:
                # Use the test configuration
                param_specified: Dict[str, float] = \
                    task.tasks_params[task_trace[0]].test_configs[test_configs_idxes[0]]
        else:
            # Child tasks exist, the final task should always be a leaf task
            child_task_trace: List[int] = task_trace[1:]
            child_test_configs_idxes: List[Optional[int]] = test_configs_idxes[1:]
            # Generate network for child tasks
            for child_task_depth in range(len(child_task_trace)):
                child_task_idx: int = child_task_trace[child_task_depth]
                child_task_params: TaskParams = child_task.tasks_params[child_task_idx]
                weights, layers_params, layers_param_shapes, global_mode, local_mode = \
                    RecursiveTask.model_params_generate(
                        hypernet_proxy=HypernetProxy(
                            global_mode=global_mode,
                            local_mode=local_mode,
                            layers_params=layers_params,
                            layers_param_shapes=layers_param_shapes,
                            structure_fourier_encoder=hypernet.structure_input_network[0],
                            index_fourier_encoder=hypernet.index_input_network[0],
                            training=training),
                        task_params=child_task_params,
                        test_config_idx=child_test_configs_idxes[child_task_depth],
                        preallocated_memory=preallocated_memory
                        )
                # Append the generated parameters for the child task
                model_params_chains.append(
                    (weights, layers_params, layers_param_shapes, global_mode, local_mode))
                # Check if reach the leaf task, if so, return the output tensor 
                #   interacting with input tensor x, otherwise, continue generating the
                #   next hypernetwork.
                if child_task_depth < len(child_task_trace) - 1:
                    # Non-leaf child tasks, update the child tasks for next generation
                    child_task = child_task.child_tasks[child_task_idx]
                else:
                    # Leaf child tasks, interact with the input tensor x 
                    #   for the final output tensor
                    if child_test_configs_idxes[child_task_depth] is None:
                        # Use the training configuration
                        param_specified: Dict[str, float] = child_task_params.param_specified
                    else:
                        # Use the test configuration
                        param_specified: Dict[str, float] = \
                            child_task_params.test_configs[child_test_configs_idxes[child_task_depth]]
        # Inference
        model_type: int = int(param_specified["model_type"])
        model_utils: type = MODEL_UTILS_DICT[model_type]
        x: torch.Tensor = model_utils.apply_weights(
            *x,
            local_mode=local_mode,
            layers_params=layers_params,
            layers_param_shapes=layers_param_shapes,
            training=training)            
        return x, global_mode, local_mode, model_params_chains
    
    @staticmethod
    def model_modularize(
        hypernet: HyperNetwork,
        task: MultiTask,
        task_trace: List[int],
        test_configs_idxes: List[Optional[int]],
        preallocated_memory: PreallocatedMemory
        ) -> Tuple[List[torch.nn.Module],
                   List[Tuple[Dict[str, float], List[Dict[str, float]]]]]:
        '''
        Modularize all the models along the trace through hypernetworks. Nitice that
            the current task should be a recursive task.
        Example 1:
        Hypernet ---> kth RecursiveTask (task_params)
                 ---> mth child RecursiveTask (child_task.task_params[m])
                 ---> nth grand RecursiveTask (child_tasks[m].child_task.task_params[n]) 
                 ---> ith leaf Task (child_tasks[m].child_tasks[n].child_task.task_params[i])
            the child_task_trace = [k, m, n, i], child_task, child_tasks[m].child_task are
            RecursiveTask, and child_tasks[m].child_tasks[n].child_task is a MultiTask.
        Example 2:
        Hypernet ---> kth RecursiveTask (task_params)
                 ---> ith leaf Task (child_task.task_params[i])
            the child_task_trace = [k, i], child_task is a MultiTask.

        Args:
            hypernet: hypernetwork.
            task: the parent task (RecursiveTask) for the model.
            task_trace: the trace (selected indexes) of child tasks. 
            test_configs_idxes: the test configuration index for each child task.
            preallocated_memory: preallocated memory for the model.
        
        Returns:
            networks_chain: the list of generated networks along the trace.
            networks_mode_chain: the list of generated modes (global_mode & local_mode) 
                along the trace.
        '''
        # Generated parameters by the hypernetwork
        networks_chain: List[torch.nn.Module] = []
        networks_mode_chain: List[Tuple[Dict[str, float], List[Dict[str, float]]]] = []
        # Calculate the weights for generated hypernetwork for child tasks
        weights, layers_params, layers_param_shapes, global_mode, local_mode = \
            Task.model_params_generate(
                hypernet=hypernet,
                task_params=task.tasks_params[task_trace[0]],
                test_config_idx=test_configs_idxes[0],
                preallocated_memory=preallocated_memory
                )
        # Modularize the first generated network
        # No child tasks, return the generated hypernetwork
        if test_configs_idxes[0] is None:
            # Use the training configuration
            param_specified: Dict[str, float] = task.tasks_params[task_trace[0]].param_specified
        else:
            # Use the test configuration
            param_specified: Dict[str, float] = \
                task.tasks_params[task_trace[0]].test_configs[test_configs_idxes[0]]
        model_type: int = int(param_specified["model_type"])
        model_cls: type[Model] = MODEL_CLS_DICT[model_type]
        network: torch.nn.Module = model_cls.modularize(
            local_mode=local_mode,
            layers_params=layers_params,
            layers_param_shapes=layers_param_shapes)
        # Append the generated hypernetwork and mode
        networks_chain.append(network)
        networks_mode_chain.append((global_mode, local_mode))

        # Generate network for child tasks
        child_task: Optional[MultiTask] = task.child_tasks[task_trace[0]]
        child_task_trace: List[int] = task_trace[1:]
        child_test_configs_idxes: List[Optional[int]] = test_configs_idxes[1:]
        for child_task_depth in range(len(child_task_trace)):
            child_task_idx: int = child_task_trace[child_task_depth]
            child_task_params: TaskParams = child_task.tasks_params[child_task_idx]
            weights, layers_params, layers_param_shapes, global_mode, local_mode = \
                RecursiveTask.model_params_generate(
                    hypernet_proxy=HypernetProxy(
                        global_mode=global_mode,
                        local_mode=local_mode,
                        layers_params=layers_params,
                        layers_param_shapes=layers_param_shapes,
                        structure_fourier_encoder=hypernet.structure_input_network[0],
                        index_fourier_encoder=hypernet.index_input_network[0],
                        training=False),
                    task_params=child_task_params,
                    test_config_idx=child_test_configs_idxes[child_task_depth],
                    preallocated_memory=preallocated_memory
                    )
            # Modularize the generated network
            if child_test_configs_idxes[child_task_depth] is None:
                # Use the training configuration
                param_specified: Dict[str, float] = child_task_params.param_specified
            else:
                # Use the test configuration
                param_specified: Dict[str, float] = \
                    child_task_params.test_configs[child_test_configs_idxes[child_task_depth]]
            model_type: int = int(param_specified["model_type"])
            model_cls: type[Model] = MODEL_CLS_DICT[model_type]
            network: torch.nn.Module = model_cls.modularize(
                local_mode=local_mode,
                layers_params=layers_params,
                layers_param_shapes=layers_param_shapes)
            # Append the generated hypernetwork and mode
            networks_chain.append(network)
            networks_mode_chain.append((global_mode, local_mode))
            # Check if reach the leaf task
            if child_task_depth < len(child_task_trace) - 1:
                # Non-leaf child tasks, and not the last task in the trace,
                #   update the child tasks for next generation
                child_task = child_task.child_tasks[child_task_idx]
        return networks_chain, networks_mode_chain

    @staticmethod
    def sample_tasks_trace(parent_task: MultiTask,
                           ) -> Tuple[List[int], MultiTask]:
        '''
        Sample a trace of tasks from the parent task and child tasks. From
            the first child of the parent task to the task index of 
            the leaf task, and the sub-task index of the leaf task.

        Args:
            parent_task: the parent task (should be a RecursiveTask).
        
        Returns:
            task_trace: the trace of tasks.
            task_params: the final task params.
        '''
        task_trace: List[int] = []
        # Sample a parent task
        current_task: MultiTask = parent_task
        while True:
            # Sample a child task or task param
            task_idx: int = torch.multinomial(torch.tensor(current_task.tasks_prob), 1).item()
            task_trace.append(task_idx)
            if isinstance(current_task, RecursiveTask):
                # Update the child task
                next_task: Optional[MultiTask] = current_task.child_tasks[task_idx]
                if next_task is not None:
                    # Update the current task
                    current_task = next_task
                else:
                    # No child tasks, break the loop
                    break
            else:
                # No child tasks, break the loop
                break
        # Retrieve the task params
        task_params: TaskParams = current_task.tasks_params[task_idx]
        return task_trace, task_params

    @staticmethod
    def get_child_task_params_from_trace(parent_task: MultiTask,
                                         task_trace: List[int]
                                         ) -> TaskParams:
        '''
        Get the leaf child task params from the trace.

        Args:
            parent_task: the parent task (should be a RecursiveTask).
            task_trace: the trace of tasks.
        
        Returns:
            current_task_params: the selected task params.
        '''
        # Get the child task from the trace
        current_task: Optional[MultiTask] = parent_task
        for child_task_idx in task_trace[:-1]:
            # Walk through the child tasks
            current_task = current_task.child_tasks[child_task_idx]
        # Get the child task params
        current_task_params: TaskParams = current_task.tasks_params[task_trace[-1]]
        return current_task_params
    
    @staticmethod
    def get_all_child_tasks_traces_inner(parent_task: Optional[MultiTask],
                                         current_trace: List[int],
                                         collected_traces: List[List[int]]
                                         ) -> None:
        '''
        Inner function of get_all_child_tasks_traces.

        Args:
            parent_task: the parent task (should be a RecursiveTask).
            current_trace: the current trace of tasks.
            collected_traces: all the collected traces of leaf child tasks.
        '''
        if parent_task is None:
            # Add the trace to the list
            collected_traces.append(current_trace.copy())
        elif isinstance(parent_task, RecursiveTask):
            # Non-leaf task, iterate over the child tasks
            for child_idx, child_task in enumerate(parent_task.child_tasks):
                # Append the current index to the trace
                current_trace.append(child_idx)
                # Recursively call the function for the child task
                RecursiveTask.get_all_child_tasks_traces_inner(
                    parent_task=child_task,
                    current_trace=current_trace,
                    collected_traces=collected_traces
                    )
                # Remove the last index from the trace
                current_trace.pop()
        else:
            # Leaf task, traverse its sub-tasks and add the trace to the list
            for param_idx in range(len(parent_task.tasks_params)):
                # Append the current index to the trace
                current_trace.append(param_idx)
                # Add the trace to the list
                collected_traces.append(current_trace.copy())
                # Remove the last index from the trace
                current_trace.pop()

    @staticmethod
    def get_all_child_tasks_traces(parent_task: MultiTask) -> List[List[int]]:
        '''
        Collect all the traces of leaf child tasks via depth-first search.

        Args:
            parent_task: the parent task (should be a RecursiveTask).

        Returns:
            child_tasks_traces: the traces of leaf child tasks.
        '''
        child_tasks_traces: List[List[int]] = []
        current_trace: List[int] = []
        # Call the inner function to collect all traces
        RecursiveTask.get_all_child_tasks_traces_inner(
            parent_task=parent_task,
            current_trace=current_trace,
            collected_traces=child_tasks_traces
            )
        return child_tasks_traces

    def initializing(self,
                     hypernet: HyperNetwork,
                     create_opt_sch_func: Callable[[int], Tuple[torch.optim.Optimizer, 
                                                      torch.optim.lr_scheduler.SequentialLR]],                    
                     steps: int,
                     update_steps: int,
                     writer: SummaryWriter,
                     log_freq: int = 100,
                     check_weight_stats: bool = False,
                     ) -> None:
        '''
        Initializing the weights generated by the hypernetwork. Notice that the whole 
            chain of the generated networks (not just the last one) are trained to match the
            desired distribution of the weights. 

        Args:
            - hypernet: hypernetwork.
            - create_opt_sch_func: function to create optimizer and scheduler every new initialization stage.
            - steps: number of steps.
            - update_steps: number of steps for increasing the range of losses update.
                For example, when step < update_steps, the loss is calculated based on 
                the init loss of first model in the model chain; otherwise when 
                step < 2 * update_steps, the loss is calculated based on the init loss of
                the second model in the model chain, and so on.
            - writer: tensorboard writer.
            - log_freq: log frequency.
            - check_weight_stats: whether to check the weight statistics during initialization.
        '''
        hypernet.train()
        for step in tqdm(range(steps), desc='Initializing'):
            # Create new optimizer and scheduler every new initialization stage
            if step % update_steps == 0:
                optimizer, scheduler = create_opt_sch_func(depth=step // update_steps)
                # Get gradient scaler
                scaler: Optional[GradScaler] = get_grad_scaler()
                # Initialize the running statistics
                running_loss: torch.Tensor = torch.tensor(0.0, device='cuda')
                running_step: int = 0

            # Sample a trace of tasks
            task_trace, _ = RecursiveTask.sample_tasks_trace(
                parent_task=self
                )
            optimizer.zero_grad()
            with get_precision_ctx():
                # Forward pass for initialization, gain the whole chain of network parameters
                model_params_chains: List[Tuple[torch.Tensor,
                            List[List[torch.Tensor]], List[List[List[int]]], 
                            Dict[str, float], List[Dict[str, float]]]] = \
                    RecursiveTask.model_params_generate_trace(
                        hypernet=hypernet,
                        task=self,
                        task_trace=task_trace,
                        test_configs_idxes=[None] * len(task_trace),
                        preallocated_memory=self.preallocated_memory,
                        training=True
                    )
                # Traverse the chain of network parameters and calculate the matching loss along the chain progressively
                losses: List[torch.Tensor] = []
                for model_params in model_params_chains:
                    local_mode: List[Dict[str, float]] = model_params[4]
                    layers_params: List[List[torch.Tensor]] = model_params[1]
                    # Calculate the target initialization statistics
                    target_layers_stats: List[List[Tuple[float, float]]] = ModelUtils.get_weights_initial_statistic(
                        local_mode=local_mode
                    )
                    # Iterate the weights and calculate the distribution matching loss
                    loss_model: torch.Tensor = Task.cal_init_loss(
                        layers_params=layers_params,
                        target_layers_stats=target_layers_stats
                    )
                    losses.append(loss_model)
                loss = losses[min(step // update_steps, len(losses) - 1)]

                if check_weight_stats:
                    # Check the weight statistics, also log all losses (the log only consider single chain for now)
                    for loss_idx, loss_node in enumerate(losses):
                        writer.add_scalar(f"Initialize/loss_{loss_idx}_step", loss_node.item(), step)
                    # Log the weight statistics of all models in the chain
                    for model_idx, model_params in enumerate(model_params_chains):
                        layers_params: List[List[torch.Tensor]] = model_params[1]
                        mean_stds_layers = Task.cal_weight_stats(layers_params)
                        for layer_idx in range(len(mean_stds_layers)):
                            for component_idx in range(len(mean_stds_layers[layer_idx])):
                                for statistic_idx in range(len(mean_stds_layers[layer_idx][component_idx])):
                                    if mean_stds_layers[layer_idx][component_idx][statistic_idx].numel() > 0:
                                        if statistic_idx == 0:
                                            stat_str = "mean"
                                        else:
                                            stat_str = "std"
                                        writer.add_scalar(f"Initialize_weight_stats/model_{model_idx}/layer_{layer_idx}/"
                                            f"component_{component_idx}/{stat_str}", mean_stds_layers[layer_idx][component_idx][statistic_idx].item(), step)

                # Update running statistics
                running_loss += loss.detach()
                running_step += 1
                # Backward pass
                if scaler is not None:
                    # Use mixed precision
                    scaler.scale(loss).backward()

                    # Clip the gradients of the hypernetwork
                    scaler.unscale_(optimizer)  # Unscale before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.01)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Full precisioin
                    loss.backward()
                    # Clip the gradients of the hypernetwork
                    grad_norm = torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.01)
                    optimizer.step()
                if check_weight_stats:
                    # Log the gradient norm
                    writer.add_scalar("Initialize/grad_norm", grad_norm.item(), step)
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
                 hypernet: HyperNetwork,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.SequentialLR,
                 steps: int,
                 writer: SummaryWriter,
                 validate_on_test_param: bool = False,
                 check_run_freq: int = 1,
                 check_freq: int = 1,
                 check_weight_stats: bool = False,
                 test: bool = False
                 ) -> List[List[Dict[str, Dict[str, float]]]]:
        '''
        Training the hypernetwork.

        Args:
            - hypernet: hypernetwork.
            - optimizer: optimizer.
            - scheduler: scheduler.
            - steps: number of steps.
            - writer: tensorboard writer.
            - validate_on_test_param: whether to validate on test parameters.
            - check_run_freq: check frequency for running statistics.
            - check_freq: check frequency.
            - check_weight_stats: whether to check the weight statistics during training.
            - test: whether to test the model on the test set after training.
        '''
        # Get gradient scaler
        scaler: Optional[GradScaler] = get_grad_scaler()
        # Initialize the running statistics
        running_loss: torch.Tensor = torch.tensor(0.0, device='cuda')
        running_corrects: torch.Tensor = torch.tensor(0, device='cuda')
        total: int = 0
        total_classify: int = 0

        # Evaluate results
        eval_results: List[List[Dict[str, Dict[str, float]]]] = []

        for step in tqdm(range(steps), desc='Training'):
            # Sample a trace of tasks
            task_trace, child_task_params = RecursiveTask.sample_tasks_trace(
                parent_task=self
                )
            # Get task type
            child_task_type: int = int(child_task_params.param_specified["task_type"])
            # Get the loss function
            child_loss_func: torch.nn.Module = child_task_params.loss_func
            # Training mode
            hypernet.train()
            # Iterate over the training data
            if child_task_type != TaskType.GRAPH_NODE_CLASSIFICATION.value:
                # Non-graph data training
                inputs, targets = next(child_task_params.train_inf_loader)
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                with get_precision_ctx():
                    # Forward pass
                    if child_task_type == TaskType.TEXT_CLASSIFICATION.value:
                        # Get padding mask for text classification
                        padding_mask = (inputs == PAD_TOKEN_IDX)
                        x = [inputs, padding_mask]
                    else:
                        x = [inputs]
                    outputs, _, _, model_params_chains = RecursiveTask.model_infer(
                        x=x,
                        hypernet=hypernet,
                        task=self,
                        task_trace=task_trace,
                        test_configs_idxes=[None] * len(task_trace),
                        preallocated_memory=self.preallocated_memory,
                        training=True
                        )
                    # Compute the loss
                    loss = child_loss_func(outputs, targets)
                    if check_weight_stats:
                        # Log the training statistics
                        writer.add_scalar('Loss_step/Train', loss.item(), step)
                        # Log the weight statistics of all models in the chain
                        for model_idx, model_params in enumerate(model_params_chains):
                            layers_params: List[List[torch.Tensor]] = model_params[1]
                            mean_stds_layers = Task.cal_weight_stats(layers_params)
                            for layer_idx in range(len(mean_stds_layers)):
                                for component_idx in range(len(mean_stds_layers[layer_idx])):
                                    for statistic_idx in range(len(mean_stds_layers[layer_idx][component_idx])):
                                        if mean_stds_layers[layer_idx][component_idx][statistic_idx].numel() > 0:
                                            if statistic_idx == 0:
                                                stat_str = "mean"
                                            else:
                                                stat_str = "std"
                                            writer.add_scalar(f"Training_weight_stats/model_{model_idx}/layer_{layer_idx}/"
                                                f"component_{component_idx}/{stat_str}", mean_stds_layers[layer_idx][component_idx][statistic_idx].item(), step)
                        # Log the outputs statistics
                        writer.add_scalar('Training_outputs/mean', outputs.mean().item(), step)
                        writer.add_scalar('Training_outputs/std', outputs.std(correction=0).item(), step)
                    # Update the running statistics
                    data_size: int = targets.size(0)
                    running_loss += loss.detach() * data_size
                    total += data_size
                    if child_task_type in [TaskType.IMAGE_CLASSIFICATION.value,
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

                    # Clip the gradients of the hypernetwork
                    scaler.unscale_(optimizer)  # Unscale before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.01)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Use full precision
                    loss.backward()
                    # Clip the gradients of the hypernetwork
                    grad_norm = torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.01)
                    optimizer.step()
            else:
                # Graph data training
                data: BaseData = child_task_params.graph_dataset[0].cuda()
                x, edge_index = data.x, data.edge_index
                optimizer.zero_grad()
                with get_precision_ctx():
                    # Forward pass
                    outputs, _, _, model_params_chains = RecursiveTask.model_infer(
                        x=[x, edge_index],
                        hypernet=hypernet,
                        task=self,
                        task_trace=task_trace,
                        test_configs_idxes=[None] * len(task_trace),
                        preallocated_memory=self.preallocated_memory,
                        training=True)
                    # Compute the loss
                    loss = child_loss_func(outputs[data.train_mask], data.y[data.train_mask])
                    if check_weight_stats:
                        # Log the training statistics
                        writer.add_scalar('Loss_step/Train', loss.item(), step)
                        # Log the weight statistics of all models in the chain
                        for model_idx, model_params in enumerate(model_params_chains):
                            layers_params: List[List[torch.Tensor]] = model_params[1]
                            mean_stds_layers = Task.cal_weight_stats(layers_params)
                            for layer_idx in range(len(mean_stds_layers)):
                                for component_idx in range(len(mean_stds_layers[layer_idx])):
                                    for statistic_idx in range(len(mean_stds_layers[layer_idx][component_idx])):
                                        if mean_stds_layers[layer_idx][component_idx][statistic_idx].numel() > 0:
                                            if statistic_idx == 0:
                                                stat_str = "mean"
                                            else:
                                                stat_str = "std"
                                            writer.add_scalar(f"Training_weight_stats/model_{model_idx}/layer_{layer_idx}/"
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

                    # Clip the gradients of the hypernetwork
                    scaler.unscale_(optimizer)  # Unscale before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.01)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Use full precision
                    loss.backward()               
                    # Clip the gradients of the hypernetwork
                    grad_norm = torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=0.01)
                    optimizer.step()
            if check_weight_stats:
                # Log the gradient norm
                writer.add_scalar("Training/grad_norm", grad_norm.item(), step)
            if (step + 1) % check_run_freq == 0 or (step + 1) == steps:
                # Log the training statistics
                avg_loss = (running_loss / total).item()
                writer.add_scalar('Loss/Train', avg_loss, step)
                # Log the learning rate
                writer.add_scalar('Learning Rate/Train', optimizer.param_groups[0]['lr'], step)                      
                print_str = f'Step {step + 1}/{steps} [Training]: Loss: {avg_loss}'
                if total_classify > 0:
                    avg_acc = (running_corrects / total_classify).item()
                    writer.add_scalar('Accuracy/Train', avg_acc, step)
                    print_str += f', Accuracy: {avg_acc}'
                print(print_str)
                # Reset the running statistics
                running_loss.fill_(0.0)
                running_corrects.fill_(0.0)
                total = 0
                total_classify = 0

            if (step + 1) % check_freq == 0 or (step + 1) == steps:
                eval_result: List[Dict[str, Dict[str, float]]] = []
                # Get all leaf child tasks and validate on them
                child_tasks_traces: List[List[int]] = \
                    RecursiveTask.get_all_child_tasks_traces(parent_task=self)
                for task_trace in child_tasks_traces:
                    eval_result_task: Dict[str, Dict[str, float]] = {}
                    # Validate on training set
                    train_result: Dict[str, float] = self.testing(hypernet=hypernet,
                        task_trace=task_trace,
                        step=step,
                        steps=steps,
                        writer=writer,
                        on_test_param=False,
                        test_split=DatasetSplit.TRAIN)[0]
                    eval_result_task[DatasetSplit.TRAIN.name] = train_result
                    if validate_on_test_param:
                        self.testing(hypernet=hypernet,
                                    task_trace=task_trace,
                                    step=step,
                                    steps=steps,
                                    writer=writer,
                                    on_test_param=True,
                                    test_split=DatasetSplit.TRAIN
                                    )             
                    # Get the leaf child task params
                    child_task_params: TaskParams = RecursiveTask.get_child_task_params_from_trace(
                        parent_task=self,
                        task_trace=task_trace
                        )             
                    if child_task_params.validate:           
                        # Validate on Train parameters
                        val_result: Dict[str, float] = self.testing(hypernet=hypernet,
                            task_trace=task_trace,
                            step=step,
                            steps=steps,
                            writer=writer,
                            on_test_param=False,
                            test_split=DatasetSplit.VAL)[0]
                        eval_result_task[DatasetSplit.VAL.name] = val_result
                    if child_task_params.validate or test:
                        # Validate on test parameters for the convenience of observation only,
                        #   do not use the test set for hyperparameter tuning
                        test_result: Dict[str, float] = self.testing(hypernet=hypernet,
                            task_trace=task_trace,
                            step=step,
                            steps=steps,
                            writer=writer,
                            on_test_param=False,
                            test_split=DatasetSplit.TEST)[0]
                        eval_result_task[DatasetSplit.TEST.name] = test_result
                    # Validate on test parameters
                    if validate_on_test_param:                 
                        if child_task_params.validate:
                            # Validate on validation set
                            self.testing(hypernet=hypernet,
                                        task_trace=task_trace,
                                        step=step,
                                        steps=steps,
                                        writer=writer,
                                        on_test_param=True,
                                        test_split=DatasetSplit.VAL,
                                        )           
                        if child_task_params.validate or test:         
                            # Validate on test set for the convenience of observation only,
                            #   do not use the test set for hyperparameter tuning
                            self.testing(hypernet=hypernet,
                                        task_trace=task_trace,
                                        step=step,
                                        steps=steps,
                                        writer=writer,
                                        on_test_param=True,
                                        test_split=DatasetSplit.TEST,
                                        )
                    eval_result.append(eval_result_task)
                eval_results.append(eval_result)
            # Update the learning rate
            scheduler.step()
        return eval_results

    def testing(self,
                hypernet: HyperNetwork,
                task_trace: List[int],
                step: int,
                steps: int,
                writer: SummaryWriter,
                on_test_param: bool = False,
                test_split: DatasetSplit = DatasetSplit.TEST,
                ) -> List[Dict[str, float]]:
        '''
        Testing on 1 task, 1 group of parameters and 1 dataset split.

        Args:
            - hypernet: hypernetwork.
            - task_trace: the trace of tasks.
            - step: current step.
            - steps: total number of steps.
            - writer: tensorboard writer.
            - on_test_param: whether to test the data on test parameters.
            - test_split: test_split, train, validation or test.
        '''
        # Get the leaf child task params
        child_task_params: TaskParams = RecursiveTask.get_child_task_params_from_trace(
            parent_task=self,
            task_trace=task_trace
            )
        if on_test_param:
            # Use the test parameters for testing on test data, notice that here we
            #   only use test configs for the leaf child task, not the whole test configs
            #   along the trace, for simplicity.
            if child_task_params.test_configs is None:
                raise ValueError("Test configs are not initialized")
            test_configs_idxes = [idx for idx in range(len(child_task_params.test_configs))]
        else:
            # Use the training parameters for testing on test data
            test_configs_idxes = [None]
        # Iterate over the test parameters
        results: List[Dict[str, float]] = []
        for test_config_idx in test_configs_idxes:
            if test_config_idx is None:
                child_param_specified: Dict[str, float] = child_task_params.param_specified
            else:
                child_param_specified: Dict[str, float] = child_task_params.test_configs[test_config_idx]
            # Get task type
            child_task_type: int = int(child_param_specified["task_type"])
            # Get the loss function
            child_loss_func: torch.nn.Module = child_task_params.loss_func
            hypernet.eval()
            # Initialize the running statistics
            running_loss = torch.tensor(0.0, device='cuda')
            running_corrects = torch.tensor(0, device='cuda')
            total = 0
            total_classify = 0
            # Logging string
            trace_str = "_".join([str(idx) for idx in task_trace])
            log_str = f'tasktrace_{trace_str}_data_{test_split.name}_param_{"train" if test_config_idx is None else f"test_{test_config_idx}"}'
            # Iterate over the validation / testing data
            with torch.no_grad():
                if child_task_type != TaskType.GRAPH_NODE_CLASSIFICATION.value:
                    # Non-graph data testing
                    if test_split == DatasetSplit.TRAIN:
                        loader = child_task_params.train_loader
                    elif test_split == DatasetSplit.VAL:
                        loader = child_task_params.val_loader
                    else:
                        loader = child_task_params.test_loader
                    for inputs, targets in tqdm(loader, desc=f'Step {step + 1}/{steps} [{log_str}]'):
                        inputs, targets = inputs.cuda(), targets.cuda()
                        with get_precision_ctx():
                            # Forward pass
                            if child_task_type == TaskType.TEXT_CLASSIFICATION.value:
                                # Get padding mask for text classification
                                padding_mask = (inputs == PAD_TOKEN_IDX)
                                x = [inputs, padding_mask]
                            else:
                                x = [inputs]
                            outputs, _, _, _ = RecursiveTask.model_infer(
                                x=x,
                                hypernet=hypernet,
                                task=self,
                                task_trace=task_trace,
                                test_configs_idxes=[None] * (len(task_trace) - 1) + [test_config_idx],
                                preallocated_memory=self.preallocated_memory,
                                training=False)
                            # Compute the loss
                            loss = child_loss_func(outputs, targets)
                            # Update the running statistics
                            data_size: int = targets.size(0)
                            running_loss += loss * data_size
                            total += data_size
                            if child_task_type in [TaskType.IMAGE_CLASSIFICATION.value,
                                            TaskType.TEXT_CLASSIFICATION.value]:
                                # Calculate the accuracy
                                _, preds = torch.max(outputs, 1)
                                corrects = torch.sum(preds == targets)
                                running_corrects += corrects
                                total_classify += data_size
                else:
                    # Graph data testing
                    data: BaseData = child_task_params.graph_dataset[0].cuda()
                    x, edge_index = data.x, data.edge_index
                    with get_precision_ctx():
                        # Forward pass
                        outputs, _, _, _ = RecursiveTask.model_infer(
                            x=[x, edge_index],
                            hypernet=hypernet,
                            task=self,
                            task_trace=task_trace,
                            test_configs_idxes=[None] * (len(task_trace) - 1) + [test_config_idx],
                            preallocated_memory=self.preallocated_memory,
                            training=False)
                        # Compute the loss
                        if test_split == DatasetSplit.TRAIN:
                            mask = data.train_mask
                        elif test_split == DatasetSplit.VAL:
                            mask = data.val_mask
                        else:
                            mask = data.test_mask
                        loss = child_loss_func(outputs[mask], data.y[mask])
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
            writer.add_scalar(f'Loss/{log_str}', avg_loss, step)
            print_str = f'Step {step + 1}/{steps} [{log_str}]: Loss: {avg_loss}'
            result: Dict[str, float] = {'loss': avg_loss}
            if total_classify > 0:
                avg_acc = (running_corrects / total_classify).item()
                writer.add_scalar(f'Accuracy/{log_str}', avg_acc, step)
                print_str += f', Accuracy: {avg_acc}'
                result['accuracy'] = avg_acc
            if child_task_type == TaskType.FORMULA_REGRESSION.value:
                # Log the root mean square error
                rmse = math.sqrt(avg_loss)
                writer.add_scalar(f'RMSE/{log_str}', rmse, step)
                print_str += f', RMSE: {rmse}'
                result['rmse'] = rmse
            print(print_str)
            results.append(result)
        return results

    def exporting(self,
                  hypernet: HyperNetwork,
                  sample_num: int = 1,
                  export_pth: bool = False,
                  export_description_file: bool = False,
                  save_dir: str = 'outputs',
                  on_test_param: bool = False
                  ) -> Tuple[List[List[List[torch.nn.Module]]],
                             List[List[List[Tuple[Dict[str, float], List[Dict[str, float]]]]]]]:
        '''
        Model exporting the whole chain of networks for each task.

        Args:
            - hypernet: hypernetwork.
            - sample_num: number of networks to be sampled for exporting
            - export_pth: whether to export to pth.
            - export_description_file: whether to export the description file.
            - save_dir: directory to save the exported models.
            - on_test_param: whether to export the model on test parameters.

        Returns:
            - networks_traces: all the networks along all the networks chains.
            - networks_description_traces: all the networks descriptions along all the networks chains.

        '''
        # Get all child tasks traces
        child_tasks_traces: List[List[int]] = \
            RecursiveTask.get_all_child_tasks_traces(parent_task=self)        

        # Networks and their descriptions for each task
        networks_traces: List[List[List[torch.nn.Module]]] = []
        networks_description_traces: List[List[List[Tuple[Dict[str, float], List[Dict[str, float]]]]]] = []
        for child_task_idx, child_task_trace in enumerate(child_tasks_traces):
            trace_str = "_".join([str(idx) for idx in child_task_trace])
            # Get the leaf child task params
            child_task_params: TaskParams = RecursiveTask.get_child_task_params_from_trace(
                parent_task=self,
                task_trace=child_task_trace
                )
            if on_test_param:
                # Use the test parameters for testing on test data
                if child_task_params.test_configs is None:
                    raise ValueError("Test configs are not initialized")
                child_test_configs_idxes = [idx for idx in range(len(child_task_params.test_configs))]
            else:
                # Use the training parameters for testing on test data
                child_test_configs_idxes = [None] * sample_num
            # Networks
            networks_trace: List[List[torch.nn.Module]] = []
            # Network descriptions, i.e., global and local structures
            networks_description_trace: List[List[Tuple[Dict[str, float], List[Dict[str, float]]]]] = []            
            # Iterate over the configs
            for sample_idx, child_test_config_idx in enumerate(child_test_configs_idxes):
                hypernet.eval()
                with torch.no_grad():
                    with get_precision_ctx():
                        # Generate the network for the whole chain
                        networks_trace_config, networks_mode_trace_config = \
                            RecursiveTask.model_modularize(
                                hypernet=hypernet,
                                task=self,
                                task_trace=child_task_trace,
                                test_configs_idxes=[None] * (len(child_task_trace) - 1) + \
                                    [child_test_config_idx],
                                preallocated_memory=self.preallocated_memory
                            )
                        # Save the network
                        networks_trace.append(networks_trace_config)
                        # Save the network description
                        networks_description_trace.append(networks_mode_trace_config)
                        # Save the network to file
                        if export_pth or export_description_file:
                            # Ensure the directory exists
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            for depth, (network, (global_mode, local_mode)) in enumerate(zip(networks_trace_config, networks_mode_trace_config)):
                                # Generate name for this network
                                network_name = f'{save_dir}/tasktrace_{trace_str}_depth_{depth}_network_{"test_" if on_test_param else ""}{sample_idx}'
                                # Save the network as pth
                                if export_pth:
                                    pth_name = f'{network_name}.pth'
                                    torch.save(network, pth_name)
                                    print(f"Network saved as [pth] {pth_name}")
                                # Save the description file
                                if export_description_file:
                                    description_name = f'{network_name}.json'
                                    with open(description_name, 'w') as f:
                                        json.dump({'global structure': global_mode,
                                                'local structure': local_mode}, 
                                                f, indent=4)
                                    print(f"Network description saved as [json] {description_name}")
            # Append the networks and their descriptions to the list
            networks_traces.append(networks_trace)
            networks_description_traces.append(networks_description_trace)
        return networks_traces, networks_description_traces
