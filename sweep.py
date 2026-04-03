'''
Hyperparameters Sweeping
'''
from task.task import DatasetSplit
from model.model.model import DatasetType, TaskType, ModelType
from copy import deepcopy
import os
import numpy as np
import subprocess
from str_helper import get_main_sweep_log_dir, get_main_log_dir, \
    get_main_multi_sweep_log_dir, get_main_multi_log_dir, \
    get_main_recursive_sweep_log_dir, get_main_recursive_log_dir
import json

def get_sweep_grid_single(
        task_type: int,
        dataset_type: int,
        model_type: int,
        mix_status: str
    ):
    '''
    Get following sweep range for single task:
        1. pioneer training epochs train_e_0 for stage one and stage two searching
        2. training learning rate train_lr
        3. training epochs train_e
        4. initialization learning rate init_lr
        5. initialization steps init_s
    '''
    if task_type == TaskType.IMAGE_CLASSIFICATION.value:
        if dataset_type == DatasetType.MNIST.value:
            if mix_status == "single":
                if model_type == ModelType.MLP.value:
                    return {
                        "train_e_0": 100,
                        "train_lr_grid": [5e-5, 1e-4, 2e-4],
                        "train_e_grid": [50, 100, 200],
                        "init_lr_grid": [5e-5, 1e-4, 2e-4],
                        "init_s_grid": [50, 100, 200],
                    }
                elif model_type == ModelType.CNN.value:
                    return {
                        "train_e_0": 50,
                        "train_lr_grid": [5e-5, 1e-4, 2e-4],
                        "train_e_grid": [30, 50, 100],
                        "init_lr_grid": [5e-5, 1e-4, 2e-4],
                        "init_s_grid": [50, 100, 200],
                    }
        elif dataset_type == DatasetType.CIFAR10.value:
            if mix_status == "single":
                return {
                    "train_e_0": 200,
                    "train_lr_grid": [5e-5, 1e-4, 2e-4],
                    "train_e_grid": [200, 400, 800],
                    "init_lr_grid": [5e-5, 1e-4, 2e-4],
                    "init_s_grid": [50, 100, 200],
                }
            elif mix_status == "mixed":
                return {
                    "train_e_0": 200,
                    "train_lr_grid": [2e-5, 5e-5, 1e-4],
                    "train_e_grid": [800, 1600, 3200],
                    "init_lr_grid": [5e-5, 1e-4, 2e-4],
                    "init_s_grid": [3200, 6400, 12800],
                }
    elif task_type == TaskType.GRAPH_NODE_CLASSIFICATION.value:
        if mix_status == "single":
            if model_type == ModelType.GCN.value:
                return {
                    "train_e_0": 200,
                    "train_lr_grid": [2e-5, 5e-5, 1e-4],
                    "train_e_grid": [100, 200, 400],
                    "init_lr_grid": [5e-5, 1e-4, 2e-4],
                    "init_s_grid": [50, 100, 200],
                }
            elif model_type == ModelType.GAT.value:
                return {
                    "train_e_0": 200,
                    "train_lr_grid": [2e-5, 5e-5, 1e-4],
                    "train_e_grid": [100, 200, 400],
                    "init_lr_grid": [5e-5, 1e-4, 2e-4],
                    "init_s_grid": [50, 100, 200],
                }
    elif task_type == TaskType.TEXT_CLASSIFICATION.value:
        if dataset_type == DatasetType.AG_NEWS.value:
            if mix_status == "single":
                return {
                    "train_e_0": 50,
                    "train_lr_grid": [5e-5, 1e-4, 2e-4],
                    "train_e_grid": [30, 50, 100],
                    "init_lr_grid": [5e-5, 1e-4, 2e-4],
                    "init_s_grid": [50, 100, 200],
                }
            elif mix_status == "mixed":
                return {
                    "train_e_0": 50,
                    "train_lr_grid": [2e-5, 5e-5, 1e-4],
                    "train_e_grid": [50, 100, 200],
                    "init_lr_grid": [5e-5, 1e-4, 2e-4],
                    "init_s_grid": [2000, 4000, 8000],
                }
        elif dataset_type == DatasetType.IMDB.value:
            if mix_status == "single":
                return {
                    "train_e_0": 100,
                    "train_lr_grid": [5e-5, 1e-4, 2e-4],
                    "train_e_grid": [50, 100, 200],
                    "init_lr_grid": [5e-5, 1e-4, 2e-4],
                    "init_s_grid": [50, 100, 200],
                }
    elif task_type == TaskType.FORMULA_REGRESSION.value:
        if mix_status == "single":
            return {
                "train_e_0": 4000,
                "train_lr_grid": [2e-5, 5e-5, 1e-4],
                "train_e_grid": [4000],
                "init_lr_grid": [5e-5, 1e-4, 2e-4],
                "init_s_grid": [50, 100, 200],
            }
    raise ValueError(f"Invalid task type {task_type} or dataset type {dataset_type}"
                     f" or model type {model_type} or mix status {mix_status}")

def get_sweep_grid_multi():
    '''
    Get following sweep range for multi task:
        1. pioneer training epochs train_e_0 for stage one and stage two searching
        2. training learning rate train_lr
        3. training epochs train_e
        4. initialization learning rate init_lr
        5. initialization steps init_s
    '''
    return {
        "train_e_0": 40000,
        "train_lr_grid": [1e-5, 2e-5, 5e-5],
        "train_e_grid": [100000, 200000, 300000],
        "init_lr_grid": [5e-5, 1e-4, 2e-4],
        "init_s_grid": [500, 1000, 2000],
    }

def get_sweep_grid_recursive():
    '''
    Get following sweep range for recursive task:
        1. pioneer training epochs train_e_0 for stage one and stage two searching
        2. training learning rate train_lr
        3. training epochs train_e
        4. initialization learning rate init_lr
        5. initialization steps init_s
    '''
    return {
        "train_e_0": 30000,
        "train_lr_grid": [2e-5, 5e-5, 1e-4],
        "train_e_grid": [15000, 30000, 60000],
        "init_lr_grid": [5e-5, 1e-4, 2e-4],
        "init_s_grid": [1000, 2000, 4000],
    }

def compare_attribute(val_attribute, best_val_attribute, comp_direct):
    if comp_direct == 'max':
        if val_attribute > best_val_attribute:
            return 'better'
        elif val_attribute < best_val_attribute:
            return 'worse'
        return 'equal'
    elif comp_direct == 'min':
        if val_attribute < best_val_attribute:
            return 'better'
        elif val_attribute > best_val_attribute:
            return 'worse'
        return 'equal'

def is_better_result(task_type, val_result, best_val_result, update_result=True):
    '''
    Check if current val result is better than the previous best val result
    '''
    if task_type != TaskType.FORMULA_REGRESSION.value:
        result_order = ['accuracy', 'std', 'train_epochs', 'train_lr', 'init_steps', 'init_lr']
        comp_direct = ['max', 'min', 'min', 'min', 'min', 'min']

    else:
        result_order = ['rmse', 'std', 'train_epochs', 'train_lr', 'init_steps', 'init_lr']
        comp_direct = ['min', 'min', 'min', 'min', 'min', 'min']        
        
    for i, key in enumerate(result_order):
        comp = compare_attribute(val_result[key], best_val_result[key], comp_direct[i])
        if comp == 'better':
            if update_result:
                # Update all keys
                for k in result_order:
                    best_val_result[k] = val_result[k]
            return True
        elif comp == 'worse':
            return False
    # If same across attributes, return false
    return False

def get_avg_result(task_type, hyperparams_result):
    '''
    Get the average validation result from a list of validation results
    '''
    val_results = hyperparams_result['val_results']
    if task_type != TaskType.FORMULA_REGRESSION.value:
        mean = sum(result['accuracy'] for result in val_results) / len(val_results)
        std = (sum((result['accuracy'] - mean) ** 2 for result in val_results) / len(val_results)) ** 0.5
        return {'accuracy': mean,
                'std': std,
                'train_epochs': hyperparams_result['train_epochs'],
                'train_lr': hyperparams_result['train_lr'],
                'init_steps': hyperparams_result['init_steps'],
                'init_lr': hyperparams_result['init_lr']}
    else:
        mean = sum(result['rmse'] for result in val_results) / len(val_results)
        std = (sum((result['rmse'] - mean) ** 2 for result in val_results) / len(val_results)) ** 0.5
        return {'rmse': mean,
                'std': std,
                'train_epochs': hyperparams_result['train_epochs'],
                'train_lr': hyperparams_result['train_lr'],
                'init_steps': hyperparams_result['init_steps'],
                'init_lr': hyperparams_result['init_lr']}

def get_val_result_single(mix_status, multi_model_mode, test_mode, sweep_result):
    '''
    Get the val result from the result of one specific sweep config
    '''
    if mix_status == 'mixed' and multi_model_mode == 'fixed':
        if test_mode == 'full_set':
            val_result = sweep_result[-1][DatasetSplit.VAL.name]
        elif test_mode == 'hold_out':
            val_result = sweep_result[-1][f"{DatasetSplit.VAL.name}_{DatasetSplit.VAL.name}"]
    else:
        val_result = sweep_result[-2][DatasetSplit.VAL.name]
    # Wrap in a form of list for consistency with multi-task sweeping
    val_result = [val_result]
    return val_result

def get_val_result_multi(sweep_result):
    '''
    Get the val result from the result of one specific sweep config
    '''
    val_result = [r[DatasetSplit.VAL.name] for r in sweep_result[-2]]
    return val_result

def assign_args(args, train_epochs, train_lr, init_steps, init_lr, validate, seed):
    '''
    Assign hyperparameters to args
    '''
    new_args = deepcopy(args)
    if hasattr(new_args, 'train_epochs'):
        # Single task case
        new_args.train_epochs = train_epochs
    elif hasattr(new_args, 'train_steps'):
        # Multi task / Recursive task case
        new_args.train_steps = train_epochs
    else:
        raise ValueError("Args must have either train_epochs or train_steps attribute")
    new_args.train_lr = train_lr
    new_args.init_steps = init_steps
    new_args.init_lr = init_lr
    # Assign validate
    new_args.validate = validate
    # Assign seed
    new_args.seed = seed
    # Disable sweeping
    new_args.sweep = False
    return new_args

def argsort(arr, key=None):
    if key is None:
        return sorted(range(len(arr)), key=lambda i: arr[i])
    else:
        return sorted(range(len(arr)), key=lambda i: key(arr[i]))
    
def mean(lst):
    return sum(lst) / len(lst)

def std(lst):
    m = mean(lst)
    return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

def borda_rank_aggregate(hyperparams_results, task_types):
    """
    hyperparams_results: list of dicts, one per candidate, same keys as above.
    Returns: avg_rank aligned with input order.
    """
    ranks = []
    for task_idx, task_type in enumerate(task_types):
        if task_type.value != TaskType.FORMULA_REGRESSION.value:
            order = argsort(hyperparams_results, key=lambda x: (-mean([val_result[task_idx]['accuracy'] for val_result in x['val_results']]),
                                                                std([val_result[task_idx]['accuracy'] for val_result in x['val_results']]),
                                                                x['train_epochs'],
                                                                x['train_lr'],
                                                                x['init_steps'],
                                                                x['init_lr']))
        else:
            order = argsort(hyperparams_results, key=lambda x: (mean([val_result[task_idx]['rmse'] for val_result in x['val_results']]),
                                                                std([val_result[task_idx]['rmse'] for val_result in x['val_results']]),
                                                                x['train_epochs'],
                                                                x['train_lr'],
                                                                x['init_steps'],
                                                                x['init_lr']))
        rank = np.empty(len(order), dtype=float)
        rank[order] = np.arange(0, len(order))
        ranks.append(rank)
    # Calculate mean and std of ranks
    ranks = np.vstack(ranks)
    avg_rank = ranks.mean(axis=0)
    std_rank = ranks.std(axis=0, ddof=0)
    return avg_rank.tolist(), std_rank.tolist()

def check_nan(hyperparams_results):
    """
    Check if any hyperparameter combination has nan validation loss for any task.
    If so, remove it from the hyperparams_results. This is not an in-place operation.
    """
    filtered_hyperparams = []
    for hyperparams_result in hyperparams_results:
        val_results = hyperparams_result['val_results']
        has_nan = False
        for val_result in val_results:
            for task_result in val_result:
                if np.isnan(task_result['loss']):
                    has_nan = True
                    break
            if has_nan:
                break
        if not has_nan:
            filtered_hyperparams.append(hyperparams_result)
    return filtered_hyperparams

def check_guardrail(hyperparams_results, task_types, guardrails):
    """
    Check if any hyperparameter combination fails the guardrail for any task.
    If so, remove it from the hyperparams_results. This is not an in-place operation.
    """
    filtered_hyperparams = []
    for hyperparams_result in hyperparams_results:
        val_results = hyperparams_result['val_results']
        fail_guardrail = False
        for task_idx, task_type in enumerate(task_types):
            if task_type.value != TaskType.FORMULA_REGRESSION.value:
                avg_accuracy = mean([val_result[task_idx]['accuracy'] for val_result in val_results])
                if avg_accuracy < guardrails[task_idx]:
                    fail_guardrail = True
                    break
            else:
                avg_rmse = mean([val_result[task_idx]['rmse'] for val_result in val_results])
                if avg_rmse > guardrails[task_idx]:
                    fail_guardrail = True
                    break
        if not fail_guardrail:
            filtered_hyperparams.append(hyperparams_result)
    return filtered_hyperparams

def args_to_cmd(args, script_name="main.py"):
    """Convert argparse args to command list"""
    cmd = ["python", script_name]
    # Iterate through all arguments
    for arg_name, arg_value in vars(args).items():
        if arg_value is None:
            continue  # Skip None values
        elif isinstance(arg_value, bool):
            if arg_value:
                cmd.append(f"--{arg_name}")  # Only add flag for True booleans
        else:
            cmd.extend([f"--{arg_name}", str(arg_value)])
    return cmd

def get_exp_result(exp_args, exp_file, get_exp_log_dir_func):
    log_dir = get_exp_log_dir_func(exp_args)
    result_path = os.path.join(log_dir, 'results.json')
    # Check if result_path exists, if so, skip running
    if os.path.exists(result_path):
        print(f"Result file {result_path} exists, skip running.")
    else:
        # Run the main file with exp_args in a subprocess
        cmd = args_to_cmd(exp_args, script_name=exp_file)
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    # Get exp_result from result_path
    with open(result_path, 'r') as f:
        exp_result = json.load(f)
    return exp_result

def sweep_multi_(args, exp_file, get_log_dir_func, get_sweep_log_dir_func, task_types, guardrails, sweep_grid, num_runs_set, get_val_result_func, run_test=True):
    '''
    For hypernetwork training, we use following four-stage sweeping:
        1. Select a small training epochs train_e0, sweep training learning rate train_lr0 under zero initialization step; if all 
            learning rates fail, set the train_lr0 to the minimum learning rate in the grid
        2. Fix the training epochs train_e0, sweep the training learning rate train_lr0 tuned from stage one, joint sweeping
            the initialization learning rate init_lr and initialization steps init_s
        3. Fix the initialization learning rate init_lr and initialization steps init_s, sweep the training learning rate 
            train_lr and training epochs train_e
        4. Get the top 3 combinations, then run extra 2 times for
            those combinations and pick the one with the best average result
    '''
    # Print the arguments
    print("Sweep Arguments:")
    print(args)

    # Store the hyperparameters-result pairs
    hyperparams_results = []

    # Five runs for Graph experiment and one runs for other tasks before final picking
    sweep_seeds = [21962, 59358, 78117, 23547, 81797, 81105, 95819, 42270, 24014, 20964, 59992, 48426, 54406, 54947, 88247, 92615, 43510, 50150, 45681, 7209]
    num_runs, num_additional_runs, num_final_test = num_runs_set

    # Stage one
    print("Stage one: Sweep train_lr0")
    for train_lr in sweep_grid["train_lr_grid"]:
        val_results = []
        for i in range(num_runs):
            sweep_args = assign_args(args, 
                train_epochs=sweep_grid["train_e_0"],
                train_lr=train_lr,
                init_steps=0,
                init_lr=0.0,
                # Always use validate
                validate=True,
                seed=sweep_seeds[i])
            sweep_result = get_exp_result(sweep_args, exp_file, get_log_dir_func)
            # Collect val results
            val_result = get_val_result_func(sweep_result)
            val_results.append(val_result)
        # Store hyperparameters and results
        hyperparams_result = {
                "train_epochs": sweep_grid["train_e_0"],
                "train_lr": train_lr,
                'init_steps': 0,
                'init_lr': 0.0,
                'val_results': val_results
            }
        hyperparams_results.append(hyperparams_result)
    # Remove settings with nan val loss
    hyperparams_results = check_nan(hyperparams_results)
    if len(hyperparams_results) == 0:
        # If all hyperparameter combinations have nan val loss, set the best_train_lr0 to the minimum learning rate in the grid
        best_train_lr0 = min(sweep_grid["train_lr_grid"])
        print(f"All hyperparameter combinations have nan val loss. Set best_train_lr0 to the minimum learning rate in the grid: {best_train_lr0}")
    else:
        # Calculate Borda count
        avg_rank, std_rank = borda_rank_aggregate(hyperparams_results, task_types)
        # Add rank info to hyperparams_results
        for i, hyperparams_result in enumerate(hyperparams_results):
            hyperparams_result['avg_rank'] = avg_rank[i]
            hyperparams_result['std_rank'] = std_rank[i]
        # Sort by avg_rank, then by std_rank, then hyperparameters
        sorted_hyperparams = sorted(hyperparams_results, key=lambda x: (x['avg_rank'], 
                                                                        x['std_rank'],
                                                                        x['train_epochs'],
                                                                        x['train_lr'],
                                                                        x['init_steps'],
                                                                        x['init_lr']), reverse=False)
        best_train_lr0 = sorted_hyperparams[0]['train_lr']
    print(f"--------------- Best train_lr0: {best_train_lr0} ----------------")

    # Stage two
    print("Stage two: Sweep init_lr and init_s")
    for init_lr in sweep_grid["init_lr_grid"]:
        for init_steps in sweep_grid["init_s_grid"]:
            val_results = []
            for i in range(num_runs):
                sweep_args = assign_args(args, 
                    train_epochs=sweep_grid["train_e_0"],
                    train_lr=best_train_lr0,
                    init_steps=init_steps,
                    init_lr=init_lr,
                    # Always use validate
                    validate=True,
                    seed=sweep_seeds[i])
                sweep_result = get_exp_result(sweep_args, exp_file, get_log_dir_func)
                # Collect val results
                val_result = get_val_result_func(sweep_result)
                val_results.append(val_result)
            # Store hyperparameters and results
            hyperparams_result = {
                    "train_epochs": sweep_grid["train_e_0"],
                    "train_lr": best_train_lr0,
                    'init_steps': init_steps,
                    'init_lr': init_lr,
                    'val_results': val_results
                }
            hyperparams_results.append(hyperparams_result)
    # Remove settings with nan val loss
    hyperparams_results = check_nan(hyperparams_results)
    if len(hyperparams_results) == 0:
        raise ValueError("All hyperparameter combinations have nan val loss. Please expand the sweep grid.")
    # Calculate Borda count
    avg_rank, std_rank = borda_rank_aggregate(hyperparams_results, task_types)
    # Add rank info to hyperparams_results
    for i, hyperparams_result in enumerate(hyperparams_results):
        hyperparams_result['avg_rank'] = avg_rank[i]
        hyperparams_result['std_rank'] = std_rank[i]
    # Sort by avg_rank, then by std_rank, then hyperparameters
    sorted_hyperparams = sorted(hyperparams_results, key=lambda x: (x['avg_rank'], 
                                                                    x['std_rank'],
                                                                    x['train_epochs'],
                                                                    x['train_lr'],
                                                                    x['init_steps'],
                                                                    x['init_lr']), reverse=False)
    best_init_lr = sorted_hyperparams[0]['init_lr']
    best_init_steps = sorted_hyperparams[0]['init_steps']
    print(f"--------------- Best init_lr: {best_init_lr}, Best init_steps: {best_init_steps} -------------")

    # Stage three
    print("Stage three: Sweep train_lr and train_e")
    for train_lr in sweep_grid["train_lr_grid"]:
        for train_epochs in sweep_grid["train_e_grid"]:
            # Skip one search if best init & lr are 0 and train lr & epochs have been searched in stage one
            if best_init_lr == 0.0 and best_init_steps == 0 and train_epochs == sweep_grid["train_e_0"] and train_lr in sweep_grid["train_lr_grid"]:
                continue
            # Skip one search if train lr and epochs are already searched in stage two
            if train_lr == best_train_lr0 and train_epochs == sweep_grid["train_e_0"]:
                continue
            val_results = []
            for i in range(num_runs):
                sweep_args = assign_args(args, 
                    train_epochs=train_epochs,
                    train_lr=train_lr,
                    init_steps=best_init_steps,
                    init_lr=best_init_lr,
                    # Always use validate
                    validate=True,
                    seed=sweep_seeds[i])
                sweep_result = get_exp_result(sweep_args, exp_file, get_log_dir_func)
                # Collect val results
                val_result = get_val_result_func(sweep_result)
                val_results.append(val_result)
            # Store hyperparameters and results
            hyperparams_result = {
                    "train_epochs": train_epochs,
                    "train_lr": train_lr,
                    'init_steps': best_init_steps,
                    'init_lr': best_init_lr,
                    'val_results': val_results
                }
            hyperparams_results.append(hyperparams_result)
    # Remove settings with nan val loss
    hyperparams_results = check_nan(hyperparams_results)
    if len(hyperparams_results) == 0:
        raise ValueError("All hyperparameter combinations have nan val loss. Please expand the sweep grid.")
    # Add guardrail and filter out bad hyperparameter combinations
    hyperparams_results = check_guardrail(hyperparams_results, task_types, guardrails)
    if len(hyperparams_results) == 0:
        raise ValueError("All hyperparameter combinations are filtered out by guardrails. Please expand the sweep grid.")
    # Calculate Borda count
    avg_rank, std_rank = borda_rank_aggregate(hyperparams_results, task_types)
    # Add rank info to hyperparams_results
    for i, hyperparams_result in enumerate(hyperparams_results):
        hyperparams_result['avg_rank'] = avg_rank[i]
        hyperparams_result['std_rank'] = std_rank[i]
    # Sort by avg_rank, then by std_rank, then hyperparameters
    sorted_hyperparams = sorted(hyperparams_results, key=lambda x: (x['avg_rank'], 
                                                                    x['std_rank'],
                                                                    x['train_epochs'],
                                                                    x['train_lr'],
                                                                    x['init_steps'],
                                                                    x['init_lr']), reverse=False)    
    best_train_lr = sorted_hyperparams[0]['train_lr']
    best_train_epochs = sorted_hyperparams[0]['train_epochs']
    print(f"--------------- Best train_lr: {best_train_lr}, Best train_epochs: {best_train_epochs} -------------")

    # Stage four
    print("Stage four: Refine the best 3 hyperparameter combinations")
    # Pick top 3 based on Borda count
    best_hyperparams = sorted_hyperparams[:3]
    print(f"Top {len(best_hyperparams)} hyperparameter combinations selected for refinement.")
    for i, hyperparams in enumerate(best_hyperparams):
        print(f"Refining hyperparameters for combination {i + 1}: {hyperparams}")
        val_results = hyperparams['val_results']
        # Run extra times for this hyperparameter combination
        for j in range(num_additional_runs):
            refine_args = assign_args(args, 
                train_epochs=hyperparams['train_epochs'],
                train_lr=hyperparams['train_lr'],
                init_steps=hyperparams['init_steps'],
                init_lr=hyperparams['init_lr'],
                # Always use validate
                validate=True,
                seed=sweep_seeds[num_runs + j])
            refine_result = get_exp_result(refine_args, exp_file, get_log_dir_func)
            val_result = get_val_result_func(refine_result)
            val_results.append(val_result)      
    # Remove settings with nan val loss
    best_hyperparams = check_nan(best_hyperparams)
    if len(best_hyperparams) == 0:
        raise ValueError("All hyperparameter combinations have nan val loss while refining. Please expand the sweep grid.")
    # Add guardrail and filter out bad hyperparameter combinations
    best_hyperparams = check_guardrail(best_hyperparams, task_types, guardrails)
    if len(best_hyperparams) == 0:
        raise ValueError("All hyperparameter combinations are filtered out by guardrails while refining. Please expand the sweep grid.")        
    # Calculate Borda count
    avg_rank, std_rank = borda_rank_aggregate(best_hyperparams, task_types)
    # Add rank info
    for i, hyperparams_result in enumerate(best_hyperparams):
        hyperparams_result['avg_rank'] = avg_rank[i]
        hyperparams_result['std_rank'] = std_rank[i]
    # Sort by avg_rank, then by std_rank, then hyperparameters
    sorted_hyperparams = sorted(best_hyperparams, key=lambda x: (x['avg_rank'], 
                                                                    x['std_rank'],
                                                                    x['train_epochs'],
                                                                    x['train_lr'],
                                                                    x['init_steps'],
                                                                    x['init_lr']), reverse=False)
    selected_hyperparam = sorted_hyperparams[0]
    print(f"--------------- Final Selected Hyperparameters: {selected_hyperparam} -------------")

    # Save sweep results
    sweep_log_dir = get_sweep_log_dir_func(args)
    log_path = os.path.join(sweep_log_dir, 'sweep.txt')
    with open(log_path, 'w') as f:
        # Write sweep grid
        f.write(f"Sweep grid: {sweep_grid}\n")
        # Write sweep result
        f.write(f"Best init_lr: {selected_hyperparam['init_lr']}, Best init_steps: {selected_hyperparam['init_steps']}\n")
        f.write(f"Best train_lr: {selected_hyperparam['train_lr']}, Best train_epochs: {selected_hyperparam['train_epochs']}\n")
    print(f"------ Sweep results saved to {log_path} -------")

    if run_test:
        print("Run test on the best hyperparameters")
        # Run test 3 times on the tuned parameters
        for i in range(num_final_test):
            test_args = assign_args(args,
                train_epochs=selected_hyperparam['train_epochs'],
                train_lr=selected_hyperparam['train_lr'],
                init_steps=selected_hyperparam['init_steps'],
                init_lr=selected_hyperparam['init_lr'],
                # Always use no validation
                validate=False,
                seed=sweep_seeds[i])
            get_exp_result(test_args, exp_file, get_log_dir_func)

    print("Sweep completed")

def sweep_single(args, run_test=True):
    # Task setting
    task_type = TaskType[args.task_type]
    dataset_type = DatasetType[args.dataset_type]
    model_type = ModelType[args.model_type]
    mix_status = args.mix_status
    multi_model_mode = args.multi_model_mode
    test_mode = args.test_mode

    task_types = [
        task_type
    ]
    # No guardrail for single task
    guardrails = [
        float('-inf') if task_type.value != TaskType.FORMULA_REGRESSION.value else float('inf')
    ]    
    # Get sweep grid
    sweep_grid = get_sweep_grid_single(
        task_type=task_type.value,
        dataset_type=dataset_type.value,
        model_type=model_type.value,
        mix_status=mix_status
    )
    # Function to get val result
    get_val_result_func = \
        lambda sweep_result: get_val_result_single(mix_status, multi_model_mode, test_mode, sweep_result)
    # Number of runs
    if task_type.value != TaskType.GRAPH_NODE_CLASSIFICATION.value:
        num_runs = 1
        num_additional_runs = 2
        num_final_test = 3
    else:
        num_runs = 5
        num_additional_runs = 5
        num_final_test = 10
    num_runs_set = [num_runs, num_additional_runs, num_final_test]
    # Call the main sweeping function
    sweep_multi_(args, 
        exp_file="main.py", get_log_dir_func=get_main_log_dir, get_sweep_log_dir_func=get_main_sweep_log_dir, 
        task_types=task_types, guardrails=guardrails, sweep_grid=sweep_grid, num_runs_set=num_runs_set, 
        get_val_result_func=get_val_result_func, run_test=run_test)

def sweep_multi(args, run_test=True):
    # Task setting
    task_types = [
        TaskType.IMAGE_CLASSIFICATION,
        TaskType.IMAGE_CLASSIFICATION,
        TaskType.GRAPH_NODE_CLASSIFICATION,
        TaskType.GRAPH_NODE_CLASSIFICATION,
        TaskType.TEXT_CLASSIFICATION,
        TaskType.FORMULA_REGRESSION
    ]
    guardrails = [
        0.95,  # MNIST
        0.85,  # CIFAR10
        0.75,  # Cora
        0.75,  # Pubmed
        0.85,  # AG_NEWS
        5e-2   # Formula
    ]
    # Get sweep grid
    sweep_grid = get_sweep_grid_multi()
    # Function to get val result
    get_val_result_func = get_val_result_multi
    # Number of runs
    num_runs = 1
    num_additional_runs = 2
    num_final_test = 3
    num_runs_set = [num_runs, num_additional_runs, num_final_test]        
    # Call the main sweeping function
    sweep_multi_(args, 
        exp_file="main_multi.py", get_log_dir_func=get_main_multi_log_dir, get_sweep_log_dir_func=get_main_multi_sweep_log_dir,
        task_types=task_types, guardrails=guardrails, sweep_grid=sweep_grid, num_runs_set=num_runs_set, 
        get_val_result_func=get_val_result_func, run_test=run_test)

def sweep_recursive(args, run_test=True):
    # Task setting
    task_types = [
        TaskType.IMAGE_CLASSIFICATION
    ]
    guardrails = [
        float('-inf')   # MNIST, no guardrail for single task
    ]
    # Get sweep grid
    sweep_grid = get_sweep_grid_recursive()
    # Function to get val result
    get_val_result_func = get_val_result_multi
    # Number of runs
    num_runs = 1
    num_additional_runs = 2
    num_final_test = 3
    num_runs_set = [num_runs, num_additional_runs, num_final_test]    
    # Call the main sweeping function
    sweep_multi_(args, 
        exp_file="main_recursive.py", get_log_dir_func=get_main_recursive_log_dir, get_sweep_log_dir_func=get_main_recursive_sweep_log_dir,
        task_types=task_types, guardrails=guardrails, sweep_grid=sweep_grid, num_runs_set=num_runs_set, 
        get_val_result_func=get_val_result_func, run_test=run_test)
