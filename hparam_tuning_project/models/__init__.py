from torchvision import models
from .pytorch_classifiers import TorchClassifier
from .simple_models import FFN, CNN1

model_registry = {
    'torch_classifier': TorchClassifier,
    'pytorch_classifier': TorchClassifier,
    'CNN1': CNN1,
    'FFN': FFN
}

