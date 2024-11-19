from hparam_tuning_project.optimization_modules.ray_optimize import run_tuner
import yaml
from ray import tune
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    config_file = args.config_file
    output_path = args.output_path
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    results = run_tuner(cfg=cfg,
                        num_samples=30,
                        search_space={'optimizer_cfg':
                                      {'args':
                                       {'lr': tune.loguniform(1e-6, 1e-1)
                                        }
                                       }
                                      }
                        )

    with open(output_path, 'w') as f:
        yaml.dump(results, f, indent=2)
