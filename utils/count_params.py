"""
Usage:
    count_params.py [-h|--help] (<DIR>...)

Options:
    -h,--help       Show this help message
    <DIR>...        Directories with subdirectory (`best_model` or `model`)
                    generated by TransQuest
"""

from typing import Dict
from pathlib import Path
from docopt import docopt

import torch


def main():
    args = docopt(__doc__)
    directories: list = args['<DIR>']

    for directory in directories:
        directory = Path(directory)

        assert directory.exists(), FileNotFoundError
        model_dir = directory / 'best_model'
        if not model_dir.exists():
            model_dir = directory / 'model'
            assert model_dir.exists(), FileNotFoundError
        model_bin = model_dir / 'pytorch_model.bin'
        assert model_bin.exists(), FileNotFoundError

        model: Dict[str, torch.Tensor] = torch.load(
            str(model_bin), map_location=torch.device('cpu')
        )
        n_params = 0
        for value in model.values():
            n_params += value.numel()

        print(f'{directory.name}:\t#parameters={n_params}')


if __name__ == '__main__':
    main()
