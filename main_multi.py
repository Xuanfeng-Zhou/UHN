import os
# Reduce memory fragment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from task.task import Task, TaskParams, DatasetSplit
from task.multi_task import MultiTask
from model.model.model import ModelType, TaskType, DatasetType
from model.hyper_network import HyperNetwork
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
import random
import json
from optimization import compile_model
import argparse
from sweep import sweep_multi
from str_helper import get_main_multi_log_dir

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
    # Disability of structure
    with_structure = not args.no_structure

    sample_times = 0
    seed = args.seed
    while True:    
        # Random Seed
        if seed is None:
            args.seed = random.randint(0, 100000)
        # Check if the seed has been used before
        log_dir = get_main_multi_log_dir(args)
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
    Task.set_random_seed(seed=seed)

    # Tasks setting
    tasks_setting = [
        {
            'task_type': TaskType.IMAGE_CLASSIFICATION.value,
            'dataset_type': DatasetType.MNIST.value,
            'model_type': ModelType.MLP.value,
            # Mix status, always use 'single' for multi-task
            'mix_status': 'single',
        },
        {
            'task_type': TaskType.IMAGE_CLASSIFICATION.value,
            'dataset_type': DatasetType.CIFAR10.value,
            'model_type': ModelType.CNN.value,
            # Mix status, always use 'single' for multi-task
            'mix_status': 'single',
            'cnn_layer_num_per_stage': 14 # For CIFAR10, use 14 layers per stage
        },
        {
            'task_type': TaskType.GRAPH_NODE_CLASSIFICATION.value,
            'dataset_type': DatasetType.CORA.value,
            'model_type': ModelType.GCN.value,
            'mix_status': 'single',
        },
        {
            'task_type': TaskType.GRAPH_NODE_CLASSIFICATION.value,
            'dataset_type': DatasetType.PUBMED.value,
            'model_type': ModelType.GAT.value,
            'mix_status': 'single',
        },
        {
            'task_type': TaskType.TEXT_CLASSIFICATION.value,
            'dataset_type': DatasetType.AG_NEWS.value,
            'model_type': ModelType.TRANSFORMER.value,
            'mix_status': 'single',
        },
        {
            'task_type': TaskType.FORMULA_REGRESSION.value,
            'dataset_type': DatasetType.SPECIAL_KV.value,
            'model_type': ModelType.KAN.value,
            'mix_status': 'single',
        }
    ]
    # Probability of each task
    tasks_prob = [
        0.08, # MNIST
        0.55, # CIFAR10
        0.04, # CORA
        0.04, # PUBMED
        0.18, # AG_NEWS
        0.11  # SPECIAL_KV
    ]

    # Task parameters
    tasks_params = []
    for task_setting in tasks_setting:
        # Configs for base network
        param_specified, param_sampled = Task.get_basenet_config(
            **task_setting
        )
        test_configs = Task.get_basenet_test_config(
            **task_setting
        )

        # Dataset
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

        # Loss function
        loss_func = Task.get_loss_function(
            task_type=task_setting['task_type'],
            dataset_type=task_setting['dataset_type']
        )

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

    # Task initialization
    task = MultiTask(
        tasks_params=tasks_params,
        tasks_prob=tasks_prob,
        )

    # HyperNetwork
    global_structure_minmax, local_structure_minmax, index_encoding_minmax = \
        task.calculate_encode_input_minmax()    
    param_hypernet = Task.get_hypernet_config()
    hypernet = HyperNetwork(
        param_dict=param_hypernet,
        global_structure_minmax=global_structure_minmax,
        local_structure_minmax=local_structure_minmax,
        index_encoding_minmax=index_encoding_minmax,
        with_structure=with_structure
    ).cuda()
    # Compile the hypernetwork for speedup
    hypernet = compile_model(hypernet)
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
        log_freq=10
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
    )

    # Testing
    eval_result= []
    if not validate:
        print("================== Testing ==================")
        # Test on each task
        for task_idx in range(len(tasks_setting)):
            # Test on train param
            result = task.testing(
                hypernet=hypernet,
                task_idx=task_idx,
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
        f.write(f"With structure: {with_structure}\n")
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
    parser.add_argument('--init_steps', type=int, default=100, help='Initialization steps')
    parser.add_argument('--train_steps', type=int, default=10000, help='Training steps')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--train_lr', type=float, default=2e-5, help='Training learning rate')
    parser.add_argument('--validate', action='store_true', help='Validate mode')
    parser.add_argument('--no_structure', action='store_true', help='Disable structure')
    parser.add_argument('--sweep', action='store_true', help='Enable sweep mode to get hyperparameter grid for single task')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')    
    args = parser.parse_args()
    # Start training
    if not args.sweep:
        main(args)
    else:
        # Sweep
        sweep_multi(args)
