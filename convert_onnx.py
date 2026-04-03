import torch
import argparse
from model.model.model import DatasetType, TaskType, ModelType
import torch.onnx
from task.task import Task
import os
from model.layer.linear_layer import LinearLayerWrapper, LinearLayer
from model.layer.conv_layer import ConvLayerWrapper, ConvLayer
from model.layer.gcn_layer import GCNLayerWrapper, GCNLayer
from model.layer.gat_layer import GATLayerWrapper, GATLayer
from model.layer.embedding_layer import EmbeddingLayerWrapper, EmbeddingLayer
from model.layer.multihead_atteention_layer import MultiheadAttentionLayerWrapper, MultiheadAttentionLayer
from model.layer.kan_layer import KANLayerWrapper, KANLayer

# TODO: Add recursive hypernetwork support

class ONNXSingleInputModule(torch.nn.Module):
    '''
    A wrapper module to facilitate ONNX export, including MLP, CNN, KAN
    '''
    def __init__(self, model):
        super(ONNXSingleInputModule, self).__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer in model.layers:
            if isinstance(layer, LinearLayer):
                self.layers.append(LinearLayerWrapper(layer))
            elif isinstance(layer, ConvLayer):
                self.layers.append(ConvLayerWrapper(layer))
            elif isinstance(layer, KANLayer):
                self.layers.append(KANLayerWrapper(layer))
            else:
                raise ValueError(f"Layer type {type(layer)} not single input compatible")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ONNXGraphModule(torch.nn.Module):
    '''
    A wrapper module to facilitate ONNX export for graph data, including GCN, GAT
    '''
    def __init__(self, model):
        super(ONNXGraphModule, self).__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer in model.layers:
            if isinstance(layer, GCNLayer):
                self.layers.append(GCNLayerWrapper(layer))
            elif isinstance(layer, GATLayer):
                self.layers.append(GATLayerWrapper(layer))
            else:
                raise ValueError(f"Layer type {type(layer)} not graph compatible")

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
    
class ONNXTransformerModule(torch.nn.Module):
    '''
    A wrapper module to facilitate ONNX export for transformer models
    '''
    def __init__(self, model):
        super(ONNXTransformerModule, self).__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for layer in model.layers:
            if isinstance(layer, EmbeddingLayer):
                self.layers.append(EmbeddingLayerWrapper(layer))
            elif isinstance(layer, MultiheadAttentionLayer):
                self.layers.append(MultiheadAttentionLayerWrapper(layer))
            elif isinstance(layer, LinearLayer):
                self.layers.append(LinearLayerWrapper(layer))
            else:
                raise ValueError(f"Layer type {type(layer)} not transformer compatible")
            
    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            if isinstance(layer, MultiheadAttentionLayerWrapper):
                x = layer(x, padding_mask)
            else:
                x = layer(x)
        return x

def ensure_directory_exists(file_path):
    """
    Ensure the directory for the given file path exists
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return file_path

def convert_to_onnx(args):
    task_type = TaskType[args.task_type]
    model_type = ModelType[args.model_type]
    pth_name = args.pth_name
    output_onnx_name = args.output_onnx_name
    ensure_directory_exists(output_onnx_name)

    # Load the trained model
    model = torch.load(pth_name)
    # If the model was compiled, we need to access the original model
    # torch.compile wraps the original model
    if hasattr(model, '_orig_mod'):
        print("Model was compiled, extracting original model...")
        model = model._orig_mod
    model = model.cuda()
    # Always save as fp32
    model = model.float()

    # Convert the model to a onnx friendly form
    if model_type.value in [ModelType.MLP.value, ModelType.CNN.value, ModelType.KAN.value]:
        model = ONNXSingleInputModule(model)
    elif model_type.value in [ModelType.GCN.value, ModelType.GAT.value]:
        model = ONNXGraphModule(model)
    elif model_type.value in [ModelType.TRANSFORMER.value]:
        model = ONNXTransformerModule(model)
            
    model.eval()  # Set to evaluation mode

    if task_type.value == TaskType.IMAGE_CLASSIFICATION.value:
        dataset_type = DatasetType[args.dataset_type]
        if dataset_type.value == DatasetType.MNIST.value:
            dummy_input = torch.randn(1, 1, 28, 28).cuda()  # Example: batch_size=1, channels=1, height=28, width=28
        elif dataset_type.value == DatasetType.CIFAR10.value:
            dummy_input = torch.randn(1, 3, 32, 32).cuda()  # Example: batch_size=1, channels=3, height=32, width=32
        else:
            raise ValueError(f"Unknown dataset: {dataset_type.value}")
        input_names = ['input']
        dynamic_axes = {'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}
    elif task_type.value == TaskType.GRAPH_NODE_CLASSIFICATION.value:
        dataset_type = DatasetType[args.dataset_type]
        datasets = Task.get_dataset(
            task_type=TaskType.GRAPH_NODE_CLASSIFICATION.value,
            param_specified={},
            dataset_type=dataset_type.value,
            batch_size=None,
            seed=None,
            validate=None
        )
        data = datasets[0].cuda()
        dummy_input = (data.x, data.edge_index)
        input_names = ['x', 'edge_index']
        dynamic_axes = None
    elif task_type.value == TaskType.TEXT_CLASSIFICATION.value:
        vocab_size = 5000
        dummy_input = torch.randint(0, vocab_size, (1, 128)).cuda()
        input_names = ['input']
        dynamic_axes = {'input': {0: 'batch_size',
                                  1: 'sequence_length'},
                        'output': {0: 'batch_size'}}
    elif task_type.value == TaskType.FORMULA_REGRESSION.value:
        dummy_input = torch.randn(1, 2).cuda()  # Example: batch_size=1, input_dim=2
        input_names = ['input']
        dynamic_axes = {'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}
    else:
        raise ValueError(f"Unsupported task type: {task_type.value}")

    # Export the model
    torch.onnx.export(
        model,                  # Model to export
        dummy_input,            # Example input
        output_onnx_name,      # Output file name
        export_params=True,     # Store trained parameters
        opset_version=17,       # ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=input_names,   # Input tensor name
        output_names=['output'], # Output tensor name
        dynamic_axes=dynamic_axes  # Dynamic axes for variable batch size
    )

    print("Model converted successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Conversion Script")
    parser.add_argument('--task_type', type=str, choices=['IMAGE_CLASSIFICATION',
        'GRAPH_NODE_CLASSIFICATION', 'TEXT_CLASSIFICATION', 'FORMULA_REGRESSION'], default='IMAGE_CLASSIFICATION',)
    parser.add_argument('--dataset_type', type=str, default=None,
                        help='Dataset to use (default: None)')
    parser.add_argument('--model_type', type=str, default='MLP', help='Model type')
    parser.add_argument('--pth_name', type=str, default='model.pth',
                        help='Path to the trained model file (default: model.pth)')
    parser.add_argument('--output_onnx_name', type=str, default='model.onnx',
                        help='Output ONNX file name (default: model.onnx)')
    args = parser.parse_args()
    convert_to_onnx(args)
