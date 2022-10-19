from .pytorch_classifiers import TorchClassifier
from .simple_models import FFN, CNN1
from .cnn import Net
model_registry = {
    'torch_classifier': TorchClassifier,
    'pytorch_classifier': TorchClassifier,
    'CNN1': CNN1,
    'CNN2': Net,
    'FFN': FFN
}
