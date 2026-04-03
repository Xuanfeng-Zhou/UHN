from .model import Model, ModelUtils, ModelType
from .cnn_model import CNNModelUtils
from .gcn_model import GCNModel, GCNModelUtils
from .gat_model import GATModel, GATModelUtils
from .transformer_model import TransformerModel, TransformerModelUtils
from .kan_model import KANModelUtils
from .recursive_model import RecursiveModel, RecursiveModelUtils
from typing import Dict

# Model class dictionary, access via "model_type" value in the model structure
MODEL_CLS_DICT: Dict[int, type[Model]] = {
    ModelType.CNN.value: Model,
    ModelType.GCN.value: GCNModel,
    ModelType.GAT.value: GATModel,
    ModelType.TRANSFORMER.value: TransformerModel,
    ModelType.KAN.value: Model,
    ModelType.RECURSIVE.value: RecursiveModel
}
# Model utils dictionary, access via "model_type" value in the model structure
MODEL_UTILS_DICT: Dict[int, type] = {
    ModelType.CNN.value: CNNModelUtils,
    ModelType.GCN.value: GCNModelUtils,
    ModelType.GAT.value: GATModelUtils,
    ModelType.TRANSFORMER.value: TransformerModelUtils,
    ModelType.KAN.value: KANModelUtils,
    ModelType.RECURSIVE.value: RecursiveModelUtils
}
