import os
# Reduce memory fragment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from task.task import Task, TaskParams, DatasetSplit
from task.recursive_task import RecursiveTask
from model.model.model import ModelType, TaskType, DatasetType
from model.hyper_network import HyperNetwork
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
import random
import json
from optimization import compile_model
import argparse
from sweep import sweep_recursive
from str_helper import get_main_recursive_log_dir

def create_recursive_task_from_setting(tasks_setting, tasks_prob, validate, seed):
    '''
    Create recursive task from setting
    '''
    tasks_params = []
    child_tasks = []
    max_task_depth = 0
    for task_setting in tasks_setting:
        # Configs for base network
        base_config_dict = {key: value for key, value in task_setting.items() 
            if key in ['task_type', 'dataset_type', 'model_type', 'mix_status', 'mix_type', 'cnn_layer_num_per_stage']}
        param_specified, param_sampled = Task.get_basenet_config(
            **base_config_dict
        )
        test_configs = Task.get_basenet_test_config(
            **base_config_dict
        )
        if 'child_tasks_setting' in task_setting:
            # Non-leaf node, no dataset and loss function
            train_inf_loader, train_loader, val_loader, test_loader = None, None, None, None
            graph_dataset = None
            loss_func = None
            # Create child tasks recursively
            child_task, child_max_task_depth = create_recursive_task_from_setting(
                tasks_setting=task_setting['child_tasks_setting'],
                tasks_prob=task_setting['child_tasks_prob'],
                validate=validate,
                seed=seed
            )
            child_tasks.append(child_task)
            task_depth = 1 + child_max_task_depth
        else:
            # Leaf node, get dataloaders and loss function
            datasets = Task.get_dataset(
                task_type=task_setting['task_type'],
                dataset_type=task_setting['dataset_type'],
                param_specified=param_specified,
                batch_size=256,
                seed=seed,
                root='./data',
                validate=validate,
            )            
            if task_setting['task_type'] == TaskType.GRAPH_NODE_CLASSIFICATION.value:
                train_inf_loader, train_loader, val_loader, test_loader = None, None, None, None
                graph_dataset = datasets
            else:
                train_inf_loader, train_loader, val_loader, test_loader = datasets
                graph_dataset = None
            loss_func = Task.get_loss_function(
                task_type=task_setting['task_type'],
                dataset_type=task_setting['dataset_type']
            )
            # No need to create child tasks for leaf nodes
            child_tasks.append(None)
            task_depth = 1
        # Create task params
        task_params = TaskParams(
            param_specified=param_specified,
            param_sampled=param_sampled,
            loss_func=loss_func,
            train_inf_loader=train_inf_loader,
            train_loader=train_loader,
            validate=validate,
            val_loader=val_loader,
            test_loader=test_loader,
            graph_dataset=graph_dataset,
            test_configs=test_configs
        )
        tasks_params.append(task_params)
        max_task_depth = max(max_task_depth, task_depth)
    # Create recursive task
    output_task = RecursiveTask(
        parent_tasks_params=tasks_params,
        child_tasks=child_tasks,
        parent_tasks_prob=tasks_prob
    )
    return output_task, max_task_depth

def main(args):
    # Print the arguments
    print("Main Arguments:")    
    print(args)

    # Task setting
    # Initialization setting
    init_steps = args.init_steps
    # Training setting
    train_steps = args.train_steps
    # Learning rate
    init_lr = args.init_lr
    train_lr = args.train_lr
    # Validate mode
    validate = args.validate
    # Check weight stats if enabled
    check_weight_stats = args.check_weight_stats
    # Ablation study mode, show test set performance
    ablation = args.ablation
    # Recursive depth
    recursive_depth = args.recursive_depth

    # Check if recursive_depth is valid
    if recursive_depth < 1:
        raise ValueError("Recursive depth should be no less than 1.")

    sample_times = 0
    seed = args.seed
    while True:    
        # Random Seed
        if seed is None:
            args.seed = random.randint(0, 100000)
        # Check if the seed has been used before
        log_dir = get_main_recursive_log_dir(args)
        export_dir = os.path.join(log_dir, 'outputs')
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        tasks_setting_path = os.path.join(checkpoint_dir, 'tasks_setting.json')    
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

    # Tasks setting
    # Construct from bottom to top
    leaf_task_setting = {
        'task_type': TaskType.IMAGE_CLASSIFICATION.value,
        'dataset_type': DatasetType.MNIST.value,
        'model_type': ModelType.MLP.value,
        'mix_status': 'single'
    }
    for _ in range(recursive_depth):
        parent_task_setting = {
            'task_type': TaskType.RECURSIVE.value,
            'dataset_type': DatasetType.RECURSIVE_IMAGE_CLASSIFICATION.value,
            'model_type': ModelType.RECURSIVE.value,
            # Mix status, always use 'single' for recursive tasks            
            'mix_status': 'single',
            'child_tasks_setting': [
                leaf_task_setting
            ],
            'child_tasks_prob': [1.0],
        }
        leaf_task_setting = parent_task_setting
    tasks_setting = [leaf_task_setting]

    # Probability of each task
    tasks_prob = [
        1.0,  # Recursive Image Classification
    ]

    # Task initialization
    task, max_task_depth = create_recursive_task_from_setting(
        tasks_setting=tasks_setting,
        tasks_prob=tasks_prob,
        validate=validate,
        seed=seed
    )
    # Calculate init update steps for each depth
    if init_steps % max_task_depth != 0:
        raise ValueError("Initialization steps should be divisible by max recursive task depth.")
    init_update_steps = init_steps // max_task_depth

    # HyperNetwork
    global_structure_minmax, local_structure_minmax, index_encoding_minmax = \
        task.calculate_encode_input_minmax()        
    param_hypernet = Task.get_hypernet_config(task_type=TaskType.RECURSIVE.value)
    hypernet = HyperNetwork(
        param_dict=param_hypernet,
        global_structure_minmax=global_structure_minmax,
        local_structure_minmax=local_structure_minmax,
        index_encoding_minmax=index_encoding_minmax
    ).cuda()
    # Compile the hypernetwork for speedup
    hypernet = compile_model(hypernet)
    # Logger
    writer = SummaryWriter(log_dir=log_dir)

    # Initializing the generated model
    print("================== Initializing ==================")
    if recursive_depth == 1:
        def create_opt_sch_func(depth):
            opt = Task.get_optimizer(
                hypernet=hypernet,
                lr=init_lr,
                betas=(0.9, 0.999)
            )
            sch = Task.get_scheduler(
                optimizer=opt,
                epochs=init_update_steps,
                warmup_epochs=0
            )
            return opt, sch
    else:
        # Use more conservative optimizer for deeper recursive model
        def create_opt_sch_func(depth):
            if depth <= 1:
                depth_lr = init_lr / 5
            elif depth == 2:
                depth_lr = init_lr / 10
            else:
                depth_lr = init_lr / 40
            opt = Task.get_optimizer(
                hypernet=hypernet,
                lr=depth_lr,
                betas=(0.9, 0.999)
            )
            sch = Task.get_scheduler(
                optimizer=opt,
                epochs=init_update_steps,
                warmup_epochs=1000
            )
            return opt, sch

    task.initializing(
        hypernet=hypernet,
        create_opt_sch_func=create_opt_sch_func,
        steps=init_steps,
        update_steps=init_update_steps,
        writer=writer,
        log_freq=10,
        check_weight_stats=check_weight_stats
    )
        
    # Training
    print("================== Training ==================")
    # Adjust learning rate based on recursive depth
    if recursive_depth == 1:
        train_lr_ = train_lr
    elif recursive_depth == 2:
        train_lr_ = train_lr / 2
    else:
        train_lr_ = train_lr / 8
    optimizer = Task.get_optimizer(
        hypernet=hypernet,
        lr=train_lr_,
        betas=(0.9, 0.999)
    )
    scheduler = Task.get_scheduler(
        optimizer=optimizer,
        epochs=train_steps,
        warmup_epochs=1000
    )
    eval_results = task.training(
        hypernet=hypernet,
        optimizer=optimizer,
        scheduler=scheduler,
        steps=train_steps,
        writer=writer,
        validate_on_test_param=False,
        check_run_freq=200,
        check_freq=2000,
        check_weight_stats=check_weight_stats,
        # Test when ablation is enabled
        test=ablation
    )

    # Testing
    eval_result= []
    if not validate and not ablation:
        print("================== Testing ==================")
        # Get all leaf child tasks and test them
        child_tasks_traces = RecursiveTask.get_all_child_tasks_traces(
            parent_task=task
        )
        for task_trace in child_tasks_traces:
            # Test on train param
            result = task.testing(
                hypernet=hypernet,
                task_trace=task_trace,
                step=train_steps,
                steps=train_steps,
                writer=writer,
                on_test_param=False,
                test_split=DatasetSplit.TEST
            )[0]
            eval_result.append({DatasetSplit.TEST.name: result})
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

    # Save the hypernetwork
    print("================== Saving Hypernetwork ==================")
    # Check if the directory exists, if not, create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Save the hypernetwork
    torch.save(hypernet.state_dict(), checkpoint_path)
    print(f"Hypernetwork saved at {checkpoint_path}")

    # Save the tasks setting and tasks prob to json file
    with open(tasks_setting_path, 'w') as f:
        json.dump({
            'tasks_setting': tasks_setting,
            'tasks_prob': tasks_prob
        }, f, indent=4)
    print(f"Tasks setting saved at {tasks_setting_path}")

    # Save a note of current parameter settings
    print("================== Saving Note ==================")
    with open(note_path, 'w') as f:
        f.write(f"Random seed: {seed}\n")
        f.write(f"Initialization steps: {init_steps}\n")
        f.write(f"Training steps: {train_steps}\n")
        f.write(f"Initial learning rate: {init_lr}\n")
        f.write(f"Training learning rate: {train_lr}\n")
        f.write(f"Validate mode: {validate}\n")
        f.write(f"Check weight stats: {check_weight_stats}\n")
        f.write(f"Ablation study mode: {ablation}\n")
        f.write(f"Recursive depth: {recursive_depth}\n")
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
    parser = argparse.ArgumentParser(description='Hypernetwork Training for Multi Task')
    parser.add_argument('--init_steps', type=int, default=2000, help='Initialization steps')
    parser.add_argument('--train_steps', type=int, default=30000, help='Training steps')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--train_lr', type=float, default=5e-5, help='Training learning rate')
    parser.add_argument('--validate', action='store_true', help='Validate mode')
    parser.add_argument('--sweep', action='store_true', help='Enable sweep mode to get hyperparameter grid for single task')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')        
    parser.add_argument('--check_weight_stats', action='store_true', help='Check weight stats')
    parser.add_argument('--ablation', action='store_true', help='Enable ablation study, and show test set performance')
    parser.add_argument('--recursive_depth', type=int, default=1, help='Max number of recursive models generated in ' \
        'a recursive chain, should be no less than 1; when 1, only one recursive model in the chain, etc.')
    args = parser.parse_args()
    # Start training
    if not args.sweep:
        main(args)
    else:
        # Sweep
       sweep_recursive(args)
