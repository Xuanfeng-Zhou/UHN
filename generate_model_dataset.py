import random
from task.task import Task
from model.model.model import ModelType, TaskType, DatasetType
import json
import os
import argparse

def main(args):
    # Task setting
    task_type = TaskType[args.task_type]
    dataset_type = DatasetType[args.dataset_type]
    model_type = ModelType[args.model_type]
    # Mix status, 'single' or 'mixed'
    mix_status = args.mix_status
    # Mix type, 'depth' or 'width' or 'depth_and_width', for CIFAR10 only
    mix_type = args.mix_type
    # Sample full number for multi-model
    sample_full_num = args.sample_full_num
    # Max sample trials for multi-model
    max_sample_trials = args.max_sample_trials
    # Sample prime number for multi-model
    sample_prime_num = args.sample_prime_num
    # Sample prime each number for multi-model
    sample_prime_each_num = args.sample_prime_each_num
    # Save directory for model datasets
    model_set_save_dir = args.model_set_save_dir

    sample_times = 0
    while True:    
        # Random Seed
        seed = args.seed if args.seed is not None else random.randint(0, 100000)
        mix_str = f'{mix_status}'
        if mix_status == 'mixed' and task_type.value == TaskType.IMAGE_CLASSIFICATION.value and \
                dataset_type.value == DatasetType.CIFAR10.value:
            mix_str += f'_{mix_type}'        
        sample_str = f'n_{sample_full_num}_p_{sample_prime_num}_pe_{sample_prime_each_num}'
        # Check if the seed has been used before
        file_name = f"{task_type.name}_{dataset_type.name}_{model_type.name}_" \
            f"{mix_str}_{sample_str}_{seed}.json"
        # Create directory if it does not exist
        if not os.path.exists(model_set_save_dir):
            os.makedirs(model_set_save_dir)
        file_path = os.path.join(model_set_save_dir, file_name)
        if not os.path.exists(file_path):
            # Check if the dataset with same name already exists
            break
        sample_times += 1
        if sample_times > 10:
            raise RuntimeError(
                f"Failed to generate a unique dataset. Tried {sample_times} times with the same parameters: "
            )
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
    )

    # Generate model dataset
    model_datasets = Task.generate_model_dataset(
        param_specified=param_specified,
        param_sampled=param_sampled,
        sample_full_num=sample_full_num,
        max_sample_trials=max_sample_trials,
        sample_prime_num=sample_prime_num,
        sample_prime_each_num=sample_prime_each_num
    )

    # Dump model datasets to json file
    with open(file_path, 'w') as f:
        json.dump(model_datasets, f, indent=4)
    print(f"Model dataset saved to {file_path}")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Hypernetwork Training for Single Task')
    parser.add_argument('--task_type', type=str, default='IMAGE_CLASSIFICATION', help='Task type')
    parser.add_argument('--dataset_type', type=str, default='MNIST', help='Dataset type')
    parser.add_argument('--model_type', type=str, default='MLP', help='Model type')
    parser.add_argument('--mix_status', type=str, default='single', help='Mix status')
    parser.add_argument('--mix_type', type=str, default='depth_and_width', help='Mix type, "depth" or "width" or "depth_and_width", for CIFAR10 only')
    parser.add_argument('--sample_full_num', type=int, default=1000, help='Sample full number for multi-model')
    parser.add_argument('--max_sample_trials', type=int, default=50000, help='Max sample trials for multi-model')
    parser.add_argument('--sample_prime_num', type=int, default=100, help='Sample prime number for multi-model')
    parser.add_argument('--sample_prime_each_num', type=int, default=50, help='Sample prime each number for multi-model')
    parser.add_argument('--model_set_save_dir', type=str, default='model_datasets', help='Directory to save model datasets')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    # Print the arguments
    print("Model dataset arguments:")
    print(args)
    # Start training
    main(args)
