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


# have a config inherit from multiple other configs
def test_config4():
    cfg = load_cfg('./training_configs/test_config4.yaml')

    assert cfg == {
        'key1': 1,
        'key2': 'a',
        'key3': {
            'subkey1': 'Goodbye',
            'subkey2': 'World',
            'subkey3': 'from test config 4'
        },
        'key4': {
            'subkey3': {
                'another_item': 'cow',
                'subsubkey1': {
                    'item': 'mooo',
                    'item2': 'fizzbuzz'
                },
                'subsubkey2': 'foo',
                'subsubkey3': 'bar',
                'subsubkey4': 'baz',
                'subsubkey5': {
                    'subsubsubkey': 'foobarbaz'
                }
            }
        }
    }


if __name__ == '__main__':
    test_config4()