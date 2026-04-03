'''
Solve multiple tasks
'''
from .task import *

class MultiTask:
    '''
    Solve multiple tasks
    '''
    def __init__(self, 
                 tasks_params: List[TaskParams],
                 tasks_prob: Optional[List[float]] = None,
                 ) -> None:
        ''''
        Initialize the multi-task class
        '''
        # Params for each task
        self.tasks_params: List[TaskParams] = tasks_params
        # Set the probability of each task
        if tasks_prob is not None:
            self.tasks_prob: List[float] = tasks_prob
        else:
            # Set the probability of each task to be equal
            self.tasks_prob: List[float] = [1.0 / len(tasks_params)] * len(tasks_params)
        # Preallocated memory for each task
        self._preallocated_memory: Optional[PreallocatedMemory] = None

    @staticmethod
    def calculate_max_memory_size(tasks_params: TaskParams
                                  ) -> Tuple[int, int, int, int, int]:
        '''
        Calculate the maximum memory size for structure and index encoding for each task.

        Returns:
            global_structure_size: the maximum memory size for global structure.
            local_structure_size: the maximum memory size for local structures.
            encode_memory_size: the maximum memory size for encoding.
            shared_element_buffer_size: the maximum memory size for shared elements.
            arange_tensor_size: the maximum memory size for arange tensor.          
        '''
        # Calculate the size of the preallocated memory for each task
        global_structure_size_max, local_structure_size_max, encode_memory_size_max, \
            shared_element_buffer_size_max, arange_tensor_size_max = 0, 0, 0, 0, 0
        for task_params in tasks_params:
            global_structure_size, local_structure_size, encode_memory_size, shared_element_buffer_size, arange_tensor_size = \
                Task.calculate_max_memory_size(task_params)
            # Update the max size for each task
            global_structure_size_max = max(global_structure_size_max, global_structure_size)
            local_structure_size_max = max(local_structure_size_max, local_structure_size)
            encode_memory_size_max = max(encode_memory_size_max, encode_memory_size)
            shared_element_buffer_size_max = max(shared_element_buffer_size_max, shared_element_buffer_size)
            arange_tensor_size_max = max(arange_tensor_size_max, arange_tensor_size)        
        return global_structure_size_max, local_structure_size_max, encode_memory_size_max, \
            shared_element_buffer_size_max, arange_tensor_size_max
    
    @property
    def preallocated_memory(self) -> PreallocatedMemory:
        '''
        Preallocated memory for each task
        '''
        if self._preallocated_memory is None:
            # Calculate the size of the preallocated memory for each task
            global_structure_size_max, local_structure_size_max, encode_memory_size_max, \
                shared_element_buffer_size_max, arange_tensor_size_max = \
                MultiTask.calculate_max_memory_size(self.tasks_params)   
            # Preallocate memory for each task
            self._preallocated_memory = PreallocatedMemory(
                global_structure_size=global_structure_size_max,
                local_structure_size=local_structure_size_max,
                encode_memory_size=encode_memory_size_max,
                shared_element_buffer_size=shared_element_buffer_size_max,
                arange_tensor_size=arange_tensor_size_max
            )
        return self._preallocated_memory

    @staticmethod
    def retrieve_encode_input_minmax(tasks_params: List[TaskParams]
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Calculate the min and max values for the input for encoding, including
            global and local structures, and the index encoding. Notice that the minmax
            should be only calculated based on the training params in case of structual leaking.

        Returns:
            - global_structure_minmax: min and max values for the global structure.
            - local_structure_minmax: min and max values for the local structure.
            - index_encoding_minmax: min and max values for the index encoding.
        '''
        global_structure_minmax, local_structure_minmax, index_encoding_minmax = \
            Task.retrieve_encode_input_minmax(task_params=tasks_params[0])
        for task_params in tasks_params[1:]:
            # Calculate the min and max values for the input for encoding for each task
            global_structure_minmax_task, local_structure_minmax_task, index_encoding_minmax_task = \
                Task.retrieve_encode_input_minmax(task_params=task_params)
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
        return MultiTask.retrieve_encode_input_minmax(tasks_params=self.tasks_params)

    def initializing(self,
                     hypernet: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler.SequentialLR,
                     steps: int,
                     writer: SummaryWriter,
                     log_freq: int = 100
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
        '''
        # Get gradient scaler
        scaler = get_grad_scaler()
        hypernet.train()
        # Initialize the running statistics
        running_loss: torch.Tensor = torch.tensor(0.0, device='cuda')
        running_step: int = 0
        for step in tqdm(range(steps), desc='Initializing'):
            # Sample a task
            task_idx = torch.multinomial(torch.tensor(self.tasks_prob), 1).item()
            optimizer.zero_grad()
            with get_precision_ctx():
                # Forward pass for initialization
                weights, layers_params, layers_param_shapes, global_mode, local_mode = \
                    Task.model_params_generate(
                        hypernet=hypernet,
                        task_params=self.tasks_params[task_idx],
                        test_config_idx=None,
                        preallocated_memory=self.preallocated_memory,
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
                 steps: int,
                 writer: SummaryWriter,
                 validate_on_test_param: bool = False,
                 check_run_freq: int = 1,
                 check_freq: int = 1,
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
            # Sample a task
            task_idx = torch.multinomial(torch.tensor(self.tasks_prob), 1).item()
            # Get the task parameters
            task_params: TaskParams = self.tasks_params[task_idx]
            # Get task type
            task_type: int = int(task_params.param_specified["task_type"])
            # Get the loss function
            loss_func: torch.nn.Module = task_params.loss_func
            # Training mode
            hypernet.train()
            # Iterate over the training data
            if task_type != TaskType.GRAPH_NODE_CLASSIFICATION.value:
                # Non-graph data training
                inputs, targets = next(task_params.train_inf_loader)
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
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
                        test_config_idx=None,
                        preallocated_memory=self.preallocated_memory,
                        training=True
                        )
                    # Compute the loss
                    loss = loss_func(outputs, targets)
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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Use full precision
                    loss.backward()
                    optimizer.step()
            else:
                # Graph data training
                data: BaseData = task_params.graph_dataset[0].cuda()
                x, edge_index = data.x, data.edge_index
                optimizer.zero_grad()
                with get_precision_ctx():
                    # Forward pass
                    outputs, _, _, _ = Task.model_infer(
                        x=[x, edge_index],
                        hypernet=hypernet,
                        task_params=task_params,
                        test_config_idx=None,
                        preallocated_memory=self.preallocated_memory,
                        training=True)
                    # Compute the loss
                    loss = loss_func(outputs[data.train_mask], data.y[data.train_mask])
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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Use full precision
                    loss.backward()
                    optimizer.step()

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
                # Validation on each task
                for task_idx, task_params in enumerate(self.tasks_params):
                    eval_result_task: Dict[str, Dict[str, float]] = {}
                    # Validate on training set
                    train_result: Dict[str, float] = self.testing(hypernet=hypernet,
                        task_idx=task_idx,
                        step=step,
                        steps=steps,
                        writer=writer,
                        on_test_param=False,
                        test_split=DatasetSplit.TRAIN)[0]
                    eval_result_task[DatasetSplit.TRAIN.name] = train_result
                    if validate_on_test_param:
                        self.testing(hypernet=hypernet,
                                    task_idx=task_idx,
                                    step=step,
                                    steps=steps,
                                    writer=writer,
                                    on_test_param=True,
                                    test_split=DatasetSplit.TRAIN)
                    if task_params.validate:
                        # Validate on Train parameters
                        val_result: Dict[str, float] = self.testing(hypernet=hypernet,
                            task_idx=task_idx,
                            step=step,
                            steps=steps,
                            writer=writer,
                            on_test_param=False,
                            test_split=DatasetSplit.VAL)[0]
                        eval_result_task[DatasetSplit.VAL.name] = val_result
                        # Validate on Test parameters for the convenience of observation only,
                        #   do not use the test set for hyperparameter tuning
                        test_result: Dict[str, float] = self.testing(hypernet=hypernet,
                            task_idx=task_idx,
                            step=step,
                            steps=steps,
                            writer=writer,
                            on_test_param=False,
                            test_split=DatasetSplit.TEST)[0]
                        eval_result_task[DatasetSplit.TEST.name] = test_result
                        # Validate on test parameters
                        if validate_on_test_param:
                            # Validate on validation set
                            self.testing(hypernet=hypernet,
                                        task_idx=task_idx,
                                        step=step,
                                        steps=steps,
                                        writer=writer,
                                        on_test_param=True,
                                        test_split=DatasetSplit.VAL,
                                        )
                            # Validate on test set, for the convenience of observation only,
                            #   do not use the test set for hyperparameter tuning
                            self.testing(hypernet=hypernet,
                                        task_idx=task_idx,
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
                hypernet: torch.nn.Module,
                task_idx: int,
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
            - task_idx: index of the task.
            - step: current step.
            - steps: total number of steps.
            - writer: tensorboard writer.
            - on_test_param: whether to test the data on test parameters.
            - test_split: test_split, train, validation or test.
        '''
        task_params: TaskParams = self.tasks_params[task_idx]
        if on_test_param:
            # Use the test parameters for testing on test data
            if task_params.test_configs is None:
                raise ValueError("Test configs are not initialized")
            test_configs_idxes = [idx for idx in range(len(task_params.test_configs))]
        else:
            # Use the training parameters for testing on test data
            test_configs_idxes = [None]
        # Iterate over the test parameters
        results: List[Dict[str, float]] = []
        for test_config_idx in test_configs_idxes:
            if test_config_idx is None:
                param_specified: Dict[str, float] = task_params.param_specified
            else:
                param_specified: Dict[str, float] = task_params.test_configs[test_config_idx]
            task_type: int = int(param_specified["task_type"])
            hypernet.eval()
            # Initialize the running statistics
            running_loss = torch.tensor(0.0, device='cuda')
            running_corrects = torch.tensor(0, device='cuda')
            total = 0
            total_classify = 0
            # Logging string
            log_str = f'task_{task_idx}_data_{test_split.name}_param_{"train" if test_config_idx is None else f"test_{test_config_idx}"}'
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
                    for inputs, targets in tqdm(loader, desc=f'Step {step + 1}/{steps} [{log_str}]'):
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
                                preallocated_memory=self.preallocated_memory,
                                training=False)
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
                            preallocated_memory=self.preallocated_memory,
                            training=False)
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
            writer.add_scalar(f'Loss/{log_str}', avg_loss, step)
            print_str = f'Step {step + 1}/{steps} [{log_str}]: Loss: {avg_loss}'
            result: Dict[str, float] = {'loss': avg_loss}
            if total_classify > 0:
                avg_acc = (running_corrects / total_classify).item()
                writer.add_scalar(f'Accuracy/{log_str}', avg_acc, step)
                print_str += f', Accuracy: {avg_acc}'
                result['accuracy'] = avg_acc
            if task_type == TaskType.FORMULA_REGRESSION.value:
                # Log the root mean square error
                rmse = math.sqrt(avg_loss)
                writer.add_scalar(f'RMSE/{log_str}', rmse, step)
                print_str += f', RMSE: {rmse}'
                result['rmse'] = rmse
            print(print_str)
            results.append(result)
        return results

    def exporting(self,
                  hypernet: torch.nn.Module,
                  sample_num: int = 1,
                  export_pth: bool = False,
                  export_description_file: bool = False,
                  save_dir: str = 'outputs',
                  on_test_param: bool = False
                  ) -> Tuple[List[List[torch.nn.Module]],
                             List[List[Tuple[Dict[str, float], List[Dict[str, float]]]]]]:
        '''
        Model exporting for each task.

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
        # Networks and their descriptions for each task
        networks_tasks: List[List[torch.nn.Module]] = []
        networks_description_tasks: List[List[Tuple[Dict[str, float], List[Dict[str, float]]]]] = []
        for task_idx in range(len(self.tasks_params)):
            task_params: TaskParams = self.tasks_params[task_idx]
            if on_test_param:
                # Use the test parameters for testing on test data
                if task_params.test_configs is None:
                    raise ValueError("Test configs are not initialized")
                test_configs_idxes = [idx for idx in range(len(task_params.test_configs))]
            else:
                # Use the training parameters for testing on test data
                test_configs_idxes = [None] * sample_num
            # Networks
            networks: List[torch.nn.Module] = []
            # Network descriptions, i.e., global and local structures
            networks_description: List[Tuple[Dict[str, float], List[Dict[str, float]]]] = []            
            # Iterate over the configs
            for sample_idx, test_config_idx in enumerate(test_configs_idxes):
                hypernet.eval()
                with torch.no_grad():
                    with get_precision_ctx():
                        # Generate the network
                        network, global_mode, local_mode = Task.model_modularize(
                            hypernet=hypernet,
                            task_params=task_params,
                            test_config_idx=test_config_idx,
                            preallocated_memory=self.preallocated_memory
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
                            network_name = f'{save_dir}/task_{task_idx}_network_{"test_" if on_test_param else ""}{sample_idx}'
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
            networks_tasks.append(networks)
            networks_description_tasks.append(networks_description)
        return networks_tasks, networks_description_tasks
