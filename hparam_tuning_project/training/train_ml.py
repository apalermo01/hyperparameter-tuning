from hparam_tuning_project.models import sklearn_registry

class MLTrainer:

    def __init__(self,
                 model_cfg,):
        if 'args' in model_cfg:
            args = model_cfg['args']
        else:
            args = dict()

        self.model = sklearn_registry[model_cfg['model_id'](**args)


    def run_training(self, X, y):
        d
