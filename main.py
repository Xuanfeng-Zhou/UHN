import os
# Reduce memory fragment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from task.task import Task, TaskParams, DatasetSplit, ModelSetParams
from model.model.model import ModelType, TaskType, DatasetType
from model.hyper_network import HyperNetwork
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
import random
from optimization import compile_model
import argparse
import json
import torchinfo
from sweep import sweep_single
from str_helper import get_main_log_dir

def main(args):
    # Print the arguments
    print("Main Arguments:")    
    print(args)

    # Task setting
    task_type = TaskType[args.task_type]
    dataset_type = DatasetType[args.dataset_type]
    model_type = ModelType[args.model_type]
    # Mix status, 'single' or 'mixed'
    mix_status = args.mix_status
    # Mix type, 'depth' or 'width' or 'depth_and_width', for CIFAR10 only
    mix_type = args.mix_type
    # CNN layer number per stage, only used for CIFAR10 CNN single model
    cnn_layer_num_per_stage = args.cnn_layer_num_per_stage
    # Multi-model mode, 'fixed' or 'on-fly'
    multi_model_mode = args.multi_model_mode
    # Test mode for multi-model, 'full_set' or 'hold_out'
    test_mode = args.test_mode
    # Sample full number for multi-model
    sample_full_num = args.sample_full_num
    # Sample prime number for multi-model
    sample_prime_num = args.sample_prime_num
    # Sample prime each number for multi-model
    sample_prime_each_num = args.sample_prime_each_num
    # Model set save directory
    model_set_save_dir = args.model_set_save_dir
    # Model set seed
    model_set_seed = args.model_set_seed
    # Initialization setting
    init_steps = args.init_steps
    # Training setting
    train_epochs = args.train_epochs
    # Learning rate
    init_lr = args.init_lr
    train_lr = args.train_lr
    # Validate mode
    validate = args.validate
    # Disability of structure
    with_structure = not args.no_structure
    # Ablation parameters
    ablation = args.ablation
    ablation_index_fourier_n_freqs = args.ablation_index_fourier_n_freqs
    ablation_block_num = args.ablation_block_num
    ablation_hidden_size = args.ablation_hidden_size
    ablation_index_encoding_type = args.ablation_index_encoding_type
    ablation_index_positional_n_freqs = args.ablation_index_positional_n_freqs
    ablation_index_positional_sigma = args.ablation_index_positional_sigma
    # Check weight stats if enabled
    check_weight_stats = args.check_weight_stats    
    grad_clip = args.grad_clip

    sample_times = 0
    seed = args.seed
    while True:    
        # Random Seed
        if seed is None:
            args.seed = random.randint(0, 100000)
        # Check if the seed has been used before
        log_dir = get_main_log_dir(args)
        export_dir = os.path.join(log_dir, 'outputs')
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        note_path = os.path.join(log_dir, 'note.txt')
        result_path = os.path.join(log_dir, 'results.json')
        if not os.path.exists(result_path):
            seed = args.seed
            break
        sample_times += 1
        if sample_times > 10:
            raise RuntimeError("Sample fails, please clean up the directory.")

    print(f"Random seed used for reproducibility: {seed}")
    # Set random seed
    Task.set_random_seed(seed=seed)

    # Configs
    param_specified, param_sampled = Task.get_basenet_config(
        task_type=task_type.value,
        dataset_type=dataset_type.value,
        model_type=model_type.value,
        mix_status=mix_status,
        mix_type=mix_type,
        cnn_layer_num_per_stage=cnn_layer_num_per_stage
    )

    # Dataset
    datasets = Task.get_dataset(
        task_type=task_type.value,
        dataset_type=dataset_type.value,
        param_specified=param_specified,
        batch_size=256,
        seed=seed,
        root='./data',
        validate=validate,
    )
    if task_type == TaskType.GRAPH_NODE_CLASSIFICATION:
        train_loader, val_loader, test_loader = None, None, None
        graph_dataset = datasets
    else:
        _, train_loader, val_loader, test_loader = datasets
        graph_dataset = None
    # Loss function
    loss_func = Task.get_loss_function(
        task_type=task_type.value,
        dataset_type=dataset_type.value
    )

    # Model set setting for multi-model
    if mix_status == 'mixed':
        if multi_model_mode == 'fixed':
            # Reconstruct the model set path
            mix_str = f'{mix_status}'
            if mix_status == 'mixed' and task_type.value == TaskType.IMAGE_CLASSIFICATION.value and \
                    dataset_type.value == DatasetType.CIFAR10.value:
                mix_str += f'_{mix_type}'        
            sample_str = f'n_{sample_full_num}_p_{sample_prime_num}_pe_{sample_prime_each_num}'
            # Check if the seed has been used before
            model_set_file_name = f"{task_type.name}_{dataset_type.name}_{model_type.name}_" \
                f"{mix_str}_{sample_str}_{model_set_seed}.json"
            model_set_file_path = os.path.join(model_set_save_dir, model_set_file_name)
            # Load json and reconstruct the model set
            if not os.path.exists(model_set_file_path):
                raise FileNotFoundError(
                    f"Model dataset file not found: {model_set_file_path}. Please generate the model dataset first."
                )
            print(f"Loading model dataset from {model_set_file_path}")
            with open(model_set_file_path, 'r') as f:            
                model_datasets = json.load(f)

            # Get model dataset loaders
            if test_mode == 'full_set':
                loader_S, loader_S_prime = Task.get_model_dataset(
                    model_datasets=model_datasets,
                    test_mode=test_mode,
                    sample_prime_each_num=None,
                    validate=validate
                )
                model_set_params = ModelSetParams(
                    multi_model_mode=multi_model_mode,
                    test_mode=test_mode,
                    loader_S=loader_S,
                    loader_S_prime=loader_S_prime
                )
            elif test_mode == 'hold_out':
                loader_S_train, loader_S_train_prime, loader_S_val_prime, loader_S_test_prime = \
                    Task.get_model_dataset(
                        model_datasets=model_datasets,
                        test_mode=test_mode,
                        sample_prime_each_num=sample_prime_each_num,
                        validate=validate
                    )
                model_set_params = ModelSetParams(
                    multi_model_mode=multi_model_mode,
                    test_mode=test_mode,
                    loader_S_train=loader_S_train,
                    loader_S_train_prime=loader_S_train_prime,
                    loader_S_val_prime=loader_S_val_prime,
                    loader_S_test_prime=loader_S_test_prime
                )
            # Set the test configs to None, as they are not needed for fixed multi-model mode
            test_configs = None
        else:
            model_set_params = ModelSetParams(
                multi_model_mode=multi_model_mode
                )
            # Retrieve selected few models for testing
            test_configs = Task.get_basenet_test_config(
                task_type=task_type.value,
                dataset_type=dataset_type.value,
                model_type=model_type.value,
                mix_status=mix_status,
                mix_type=mix_type,
                cnn_layer_num_per_stage=cnn_layer_num_per_stage
            )    
    else:
        model_set_params = None
        test_configs = Task.get_basenet_test_config(
            task_type=task_type.value,
            dataset_type=dataset_type.value,
            model_type=model_type.value,
            mix_status=mix_status,
            mix_type=mix_type,
            cnn_layer_num_per_stage=cnn_layer_num_per_stage
        )
            
    # Task initialization
    task_params = TaskParams(
        param_specified=param_specified,
        param_sampled=param_sampled,
        loss_func=loss_func,
        train_inf_loader=None,
        train_loader=train_loader,
        validate=validate,
        val_loader=val_loader,
        test_loader=test_loader,
        graph_dataset=graph_dataset,
        test_configs=test_configs,
        model_set_params=model_set_params,
    )
    task = Task(task_params=task_params)

    # HyperNetwork
    global_structure_minmax, local_structure_minmax, index_encoding_minmax = \
        task.calculate_encode_input_minmax()
    param_hypernet = Task.get_hypernet_config(
        task_type=task_type,
        dataset_type=dataset_type,
        mix_status=mix_status,
        ablation_index_fourier_n_freqs=ablation_index_fourier_n_freqs,
        ablation_block_num=ablation_block_num,
        ablation_hidden_size=ablation_hidden_size,
        ablation_index_encoding_type=ablation_index_encoding_type,
        ablation_index_positional_n_freqs=ablation_index_positional_n_freqs,
        ablation_index_positional_sigma=ablation_index_positional_sigma
    )
    hypernet = HyperNetwork(
        param_dict=param_hypernet,
        global_structure_minmax=global_structure_minmax,
        local_structure_minmax=local_structure_minmax,
        index_encoding_minmax=index_encoding_minmax,
        with_structure=with_structure
    ).cuda()
    # Compile the hypernetwork for speedup
    hypernet = compile_model(hypernet)
    # Print the hypernetwork structure
    torchinfo.summary(hypernet)
    # Logger
    writer = SummaryWriter(log_dir=log_dir)

    # Initializing the generated model
    print("================== Initializing ==================")
    optimizer = Task.get_optimizer(
        hypernet=hypernet,
        lr=init_lr,
        betas=(0.9, 0.999)
    )
    scheduler = Task.get_scheduler(
        optimizer=optimizer,
        epochs=init_steps,
        warmup_epochs=0
    )
    task.initializing(
        hypernet=hypernet,
        optimizer=optimizer,
        scheduler=scheduler,
        steps=init_steps,
        writer=writer,
        log_freq=10,
        check_weight_stats=check_weight_stats
    )
        
    # Training
    print("================== Training ==================")
    optimizer = Task.get_optimizer(
        hypernet=hypernet,
        lr=train_lr,
        betas=(0.9, 0.999)
    )
    scheduler = Task.get_scheduler(
        optimizer=optimizer,
        epochs=train_epochs,
        warmup_epochs=5
    )
    validate_on_test_param = True if (mix_status == 'mixed' and multi_model_mode != 'fixed') else False
    eval_results = task.training(
        hypernet=hypernet,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=train_epochs,
        writer=writer,
        validate_on_test_param=validate_on_test_param,
        check_freq=10,
        check_weight_stats=check_weight_stats,
        # Test when ablation is enabled
        test=ablation,
        grad_clip=grad_clip
    )

    # Testing
    eval_result= {}
    if mix_status == 'mixed' and multi_model_mode == 'fixed':
        # Test on fixed prime model set
        if test_mode == 'full_set':
            test_data_splits = []
            if validate:
                test_data_splits.append(DatasetSplit.VAL)
            else:
                test_data_splits.append(DatasetSplit.TRAIN)
                test_data_splits.append(DatasetSplit.TEST)
            # Test on datasets
            for test_data_split in test_data_splits:
                result = task.test_model_set(
                    hypernet=hypernet,
                    epoch=train_epochs,
                    epochs=train_epochs,
                    writer=writer,
                    test_data_split=test_data_split,
                    test_model_split=None
                )
                eval_result[test_data_split.name] = result
        elif test_mode == 'hold_out':
            test_data_splits = []
            test_model_splits = [DatasetSplit.TRAIN]
            if validate:
                # Dataset
                test_data_splits.append(DatasetSplit.VAL)
                # Modelset
                test_model_splits.append(DatasetSplit.VAL)
            else:
                # Dataset
                test_data_splits.append(DatasetSplit.TRAIN)
                test_data_splits.append(DatasetSplit.TEST)
                # Modelset
                test_model_splits.append(DatasetSplit.TEST)
            # Test on different splits of the model set and data set
            for test_data_split in test_data_splits:
                for test_model_split in test_model_splits:        
                    result = task.test_model_set(
                        hypernet=hypernet,
                        epoch=train_epochs,
                        epochs=train_epochs,
                        writer=writer,
                        test_data_split=test_data_split,
                        test_model_split=test_model_split
                    )
                    eval_result[f"{test_data_split.name}_{test_model_split.name}"] = result
        else:
            raise ValueError(f"Invalid test mode: {test_mode}")
    else:
        # Test on on-the-fly sampled and predefined model structures
        if not validate and not ablation:
            print("================== Testing ==================")
            # Test on train param
            result = task.testing(
                hypernet=hypernet,
                epoch=train_epochs,
                epochs=train_epochs,
                writer=writer,
                on_test_param=False,
                test_split=DatasetSplit.TEST
            )[0]
            eval_result[DatasetSplit.TEST.name] = result
            if validate_on_test_param:
                # Test on test param
                task.testing(
                    hypernet=hypernet,
                    epoch=train_epochs,
                    epochs=train_epochs,
                    writer=writer,
                    on_test_param=True,
                    test_split=DatasetSplit.TEST
                )
    eval_results.append(eval_result)

    # Close the logger
    writer.close()

    # Exporting the generated model
    print("================== Exporting ==================")
    # Export on train param
    task.exporting(
        hypernet=hypernet,
        sample_num=1,
        export_pth=True,
        export_description_file=True,
        save_dir=export_dir,
        on_test_param=False
    )
    if validate_on_test_param:
        # Export on test param
        task.exporting(
            hypernet=hypernet,
            sample_num=1,
            export_pth=True,
            export_description_file=True,
            save_dir=export_dir,
            on_test_param=True
        )

    # Save the hypernetwork
    print("================== Saving Hypernetwork ==================")
    # Check if the directory exists, if not, create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Save the hypernetwork
    torch.save(hypernet.state_dict(), checkpoint_path)
    print(f"Hypernetwork saved at {checkpoint_path}")

    # Save a note of current parameter settings
    print("================== Saving Note ==================")
    with open(note_path, 'w') as f:
        f.write(f"Task: {task_type.name}\n")
        f.write(f"Dataset: {dataset_type.name}\n")
        f.write(f"Model: {model_type.name}\n")
        f.write(f"Mix status: {mix_status}\n")
        if mix_status == 'mixed' and task_type.value == TaskType.IMAGE_CLASSIFICATION.value and \
                dataset_type.value == DatasetType.CIFAR10.value:
            f.write(f"Mix type: {mix_type}\n")
        if mix_status == 'mixed':
            f.write(f"Multi-model mode: {multi_model_mode}\n")
            if multi_model_mode == 'fixed':
                f.write(f"Sample full number: {sample_full_num}\n")
                f.write(f"Test mode: {test_mode}\n")
                f.write(f"Sample prime number: {sample_prime_num}\n")
                f.write(f"Sample prime each number: {sample_prime_each_num}\n")
                f.write(f"Model set save directory: {model_set_save_dir}\n")
                f.write(f"Model set seed: {model_set_seed}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Initialization steps: {init_steps}\n")
        f.write(f"Training epochs: {train_epochs}\n")
        f.write(f"Initial learning rate: {init_lr}\n")
        f.write(f"Training learning rate: {train_lr}\n")
        f.write(f"Validate mode: {validate}\n")
        f.write(f"With structure: {with_structure}\n")
        f.write(f"Ablation: {ablation}\n")
        f.write(f"Ablation index for Fourier n_freqs: {ablation_index_fourier_n_freqs}\n")
        f.write(f"Ablation block number: {ablation_block_num}\n")
        f.write(f"Ablation hidden size: {ablation_hidden_size}\n")
        f.write(f"Ablation index encoding type: {ablation_index_encoding_type}\n")
        f.write(f"Ablation index positional n_freqs: {ablation_index_positional_n_freqs}\n")
        f.write(f"Ablation index positional sigma: {ablation_index_positional_sigma}\n")
        f.write(f"Check weight stats: {check_weight_stats}\n")
        f.write(f"Gradient clipping: {grad_clip}\n")
    print(f"Note saved at {note_path}")

    # Save results to json
    print("================== Saving Results ==================")
    with open(result_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"Results saved at {result_path}")

    # Ending
    print("================== End of Task ==================")
    print("Task finished!")

    return eval_results

if __name__ == '__main__':
    # Startup settings
    multiprocessing.set_start_method('spawn')

    # Parse arguments
    parser = argparse.ArgumentParser(description='Hypernetwork Training for Single Task')
    parser.add_argument('--task_type', type=str, default='IMAGE_CLASSIFICATION', help='Task type')
    parser.add_argument('--dataset_type', type=str, default='MNIST', help='Dataset type')
    parser.add_argument('--model_type', type=str, default='MLP', help='Model type')
    parser.add_argument('--mix_status', type=str, default='single', help='Mix status')
    parser.add_argument('--mix_type', type=str, default='depth_and_width', help='Mix type, "depth" or "width" or "depth_and_width", ' \
        'for CIFAR10 CNN multi model only')
    parser.add_argument('--cnn_layer_num_per_stage', type=int, default=6, help='CNN layer number per stage, only used for CIFAR10 CNN single model')
    parser.add_argument('--multi_model_mode', type=str, default='on-fly', help='Multi-model mode, "fixed" or "on-fly"')
    parser.add_argument('--test_mode', type=str, default='full_set', help='Test mode for multi-model, "full_set" or "hold_out"')
    parser.add_argument('--sample_full_num', type=int, default=1000, help='Sample full number for multi-model')
    parser.add_argument('--sample_prime_num', type=int, default=100, help='Sample prime number for multi-model')
    parser.add_argument('--sample_prime_each_num', type=int, default=50, help='Sample prime each number for multi-model')
    parser.add_argument('--model_set_save_dir', type=str, default='model_datasets', help='Model set save directory')
    parser.add_argument('--model_set_seed', type=int, default=0, help='Model set seed for reproducibility')
    parser.add_argument('--init_steps', type=int, default=100, help='Initialization steps')
    parser.add_argument('--train_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--train_lr', type=float, default=1e-4, help='Training learning rate')
    parser.add_argument('--validate', action='store_true', help='Validate mode')
    parser.add_argument('--no_structure', action='store_true', help='Disable structure')
    parser.add_argument('--ablation', action='store_true', help='Enable ablation study, and show test set performance')
    parser.add_argument('--ablation_index_fourier_n_freqs', type=int, default=None, help='Ablation index for Fourier n_freqs')
    parser.add_argument('--ablation_block_num', type=int, default=None, help='Ablation block number')
    parser.add_argument('--ablation_hidden_size', type=int, default=None, help='Ablation hidden size')
    parser.add_argument('--ablation_index_encoding_type', type=str, default=None, help='Ablation index encoding type')
    parser.add_argument('--ablation_index_positional_n_freqs', type=int, default=None, help='Ablation index positional n_freqs')
    parser.add_argument('--ablation_index_positional_sigma', type=float, default=None, help='Ablation index positional sigma')
    parser.add_argument('--sweep', action='store_true', help='Enable sweep mode to get hyperparameter grid for single task')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--check_weight_stats', action='store_true', help='Check weight stats')
    parser.add_argument('--grad_clip', action='store_true', help='Enable gradient clipping during training')
    args = parser.parse_args()
    # Start training
    if not args.sweep:
        main(args)
    else:
        # Sweep
       sweep_single(args)
