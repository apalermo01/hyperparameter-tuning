from hparam_tuning_project.utils import load_cfg
import pytest


def test_config1():

    cfg = load_cfg('./training_configs/test_config1.yaml')

    assert cfg == {
        'key1': 1,
        'key2': 'a',
        'key3': {
            'subkey1': 'Hello',
            'subkey2': 'World',
        },
        'key4': {
            'subkey3': {
                'subsubkey1': {
                    'item': 'goodbye',
                }
            }
        }
    }


def test_config2():

    cfg = load_cfg('./training_configs/test_config2.yaml')
    assert cfg == {
        'key1': 1,
        'key2': 'a',
        'key3': {
            'subkey1': 'Hello',
            'subkey2': 'World',
        },
        'key4': {
            'subkey3': {
                'subsubkey1': {
                    'item': 'goodbye',
                }
            }
        }
    }


def test_config3():
    cfg = load_cfg('./training_configs/test_config3.yaml')

    assert cfg == {
        'key1': 1,
        'key2': 'a',
        'key3': {
            'subkey1': 'Goodbye',
            'subkey2': 'World',
        },
        'key4': {
            'subkey3': {
                'another_item': 'cow',
                'subsubkey1': {
                    'item': 'goodbye',
                }
            }
        }
    }
