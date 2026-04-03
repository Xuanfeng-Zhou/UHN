'''
This script controls the switching of the JIT compiler in PyTorch.
'''
import torch
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# Set to True to enable JIT, False to disable (torch.jit.script)
enable_jit = True  
# Set to True to enable model compilation, False to disable (torch.compile)
enable_model_compile = True
# Set to True to enable mixed precision, False to disable (to float16)
enable_mixed_precision = True  

def jit_script(func_or_class):
    '''
    Scripts a function or class with JIT if enabled.
    '''
    return torch.jit.script(func_or_class) if enable_jit else func_or_class

def compile_model(model):
    '''
    Compiles a model with torch.compile if enabled.
    '''
    if enable_model_compile:
        torch.set_float32_matmul_precision('high')
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        model = torch.compile(model, mode='max-autotune', dynamic=True)
    return model

def get_grad_scaler():
    '''
    Returns a GradScaler if mixed precision is enabled, otherwise None.
    '''
    return GradScaler("cuda") if enable_mixed_precision else None

def get_precision_ctx():
    '''
    Returns the precision context for mixed precision if enabled, otherwise nullcontext.
    '''
    return autocast(device_type="cuda", dtype=torch.float16) \
        if enable_mixed_precision else nullcontext()
