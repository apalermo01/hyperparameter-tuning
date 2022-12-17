from .pytorch_classifiers import TorchClassifier
from .simple_models import FFN, CNN1
from .cnn import Net
from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors

model_registry = {
    'torch_classifier': TorchClassifier,
    'pytorch_classifier': TorchClassifier,
    'CNN1': CNN1,
    'CNN2': Net,
    'FFN': FFN
}

sklearn_registry = {
    'linear_regression': lm.LinearRegression,
    'ridge_regression': lm.Ridge,
    'lasso_regression': lm.Lasso,
    'elastic_net': lm.ElasticNet,
    'decision_tree_regression': tree.DecisionTreeRegressor,
    'random_forest_regression': ensemble.RandomForestRegressor,
    'knn_reg': neighbors.KNeighborsRegressor,

    'logistic_regression': lm.LogisticRegression,
    'knn_class': neighbors.KNeighborsClassifier,
    'decision_tree_classification': tree.DecisionTreeClassifier,
    'random_forest_classification': ensemble.RandomForestClassifier,
}
