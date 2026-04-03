from model.model.model import ModelType, TaskType, DatasetType
import os

def get_main_sweep_log_dir(args):
    task_type = TaskType[args.task_type]
    dataset_type = DatasetType[args.dataset_type]
    model_type = ModelType[args.model_type]
    # Mix status, 'single' or 'mixed'
    mix_status = args.mix_status
    # Mix type, 'depth' or 'width' or 'depth_and_width', for CIFAR10 only
    mix_type = args.mix_type
    # CNN layer number per stage, only used for CIFAR10 CNN single model
    cnn_layer_num_per_stage = args.cnn_layer_num_per_stage
    # Disability of structure
    with_structure = not args.no_structure

    ablation_index_fourier_n_freqs = args.ablation_index_fourier_n_freqs
    ablation_block_num = args.ablation_block_num
    ablation_hidden_size = args.ablation_hidden_size
    ablation_index_encoding_type = args.ablation_index_encoding_type
    ablation_index_positional_n_freqs = args.ablation_index_positional_n_freqs
    ablation_index_positional_sigma = args.ablation_index_positional_sigma

    ablation_str = '_no_struct' if not with_structure else ''
    ablation_str += f'_nfreqs_{ablation_index_fourier_n_freqs}' if ablation_index_fourier_n_freqs is not None else ''
    ablation_str += f'_blocks_{ablation_block_num}' if ablation_block_num is not None else ''
    ablation_str += f'_hidden_{ablation_hidden_size}' if ablation_hidden_size is not None else ''
    ablation_str += f'_enctype_{ablation_index_encoding_type}' if ablation_index_encoding_type is not None else ''
    ablation_str += f'_posnfreqs_{ablation_index_positional_n_freqs}' if ablation_index_positional_n_freqs is not None else ''
    ablation_str += f'_possigma_{ablation_index_positional_sigma}' if ablation_index_positional_sigma is not None else ''
    mix_str = f'{mix_status}'
    if mix_status == 'mixed' and task_type.value == TaskType.IMAGE_CLASSIFICATION.value and \
            dataset_type.value == DatasetType.CIFAR10.value:
        mix_str += f'_{mix_type}'
    if mix_status == 'single' and task_type.value == TaskType.IMAGE_CLASSIFICATION.value and \
            dataset_type.value == DatasetType.CIFAR10.value:
        mix_str += f'_nlayers_{cnn_layer_num_per_stage}'
    sweep_log_dir = f"{task_type.name}_{dataset_type.name}_{model_type.name}_" \
        f"{mix_str}{ablation_str}"
    sweep_log_dir = os.path.join('runs', sweep_log_dir)
    return sweep_log_dir

def get_main_log_dir(args):
    seed = args.seed
    # Initialization setting
    init_steps = args.init_steps
    # Training setting
    train_epochs = args.train_epochs
    # Learning rate
    init_lr = args.init_lr
    train_lr = args.train_lr
    # Validate mode
    validate = args.validate
    grad_clip = args.grad_clip

    grad_clip_str = '_gradclip' if grad_clip else ''
    validate_str = '_val' if validate else ''
    sweep_comment = f"trainlr_{train_lr}_traine_{train_epochs}_initlr_{init_lr}_inits_{init_steps}"
    sweep_log_dir = get_main_sweep_log_dir(args)
    log_dir = os.path.join(sweep_log_dir, f"{sweep_comment}{grad_clip_str}{validate_str}", f'{seed}')
    return log_dir

def get_main_multi_sweep_log_dir(args):
    # Disability of structure
    with_structure = not args.no_structure

    ablation_str = '_no_structure' if not with_structure else ''
    sweep_log_dir = f"multitask{ablation_str}"
    sweep_log_dir = os.path.join('runs', sweep_log_dir)
    return sweep_log_dir

def get_main_multi_log_dir(args):
    seed = args.seed
    # Initialization setting
    init_steps = args.init_steps
    # Training setting
    train_steps = args.train_steps
    # Learning rate
    init_lr = args.init_lr
    train_lr = args.train_lr
    # Validate mode
    validate = args.validate

    validate_str = '_val' if validate else ''
    sweep_comment = f"trainlr_{train_lr}_trains_{train_steps}_initlr_{init_lr}_inits_{init_steps}"
    sweep_log_dir = get_main_multi_sweep_log_dir(args)
    log_dir = os.path.join(sweep_log_dir, f"{sweep_comment}{validate_str}", f'{seed}')
    return log_dir

def get_main_recursive_sweep_log_dir(args):
    sweep_log_dir = os.path.join('runs', 'recursivetask')
    return sweep_log_dir

def get_main_recursive_log_dir(args):
    seed = args.seed
    # Initialization setting
    init_steps = args.init_steps
    # Training setting
    train_steps = args.train_steps
    # Learning rate
    init_lr = args.init_lr
    train_lr = args.train_lr
    # Validate mode
    validate = args.validate
    # Recursive depth
    recursive_depth = args.recursive_depth

    validate_str = '_val' if validate else ''
    recursive_depth_str ='' if recursive_depth == 1 else f'_depth_{recursive_depth}'
    sweep_comment = f"trainlr_{train_lr}_trains_{train_steps}_initlr_{init_lr}_inits_{init_steps}"
    sweep_log_dir = get_main_recursive_sweep_log_dir(args)
    log_dir = os.path.join(sweep_log_dir, f"{sweep_comment}{recursive_depth_str}{validate_str}", f'{seed}')
    return log_dir
