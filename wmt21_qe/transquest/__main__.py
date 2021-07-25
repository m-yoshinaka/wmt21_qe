from pathlib import Path
from datetime import datetime
from logging import getLogger
from typing import List

from docopt import docopt
from torch import manual_seed

from wmt21_qe.__main__ import __doc__ as docstring
from wmt21_qe import set_logger, build_dataset, save_results
from wmt21_qe.transquest.run import run_monotransquest
from wmt21_qe.transquest.config import (
    SEED, SUBMISSION_FILE, MODEL_TYPE, MODEL_NAME, mono_config
)


manual_seed(SEED)
date = datetime.now().strftime('%Y%m%d-%H%M%S')
logger = getLogger('WMT21QE')


def main():
    args = docopt(docstring)

    output_dir = args['--out-dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set logger
    running = 'train' if args['train'] else 'eval'
    log_file = output_dir + f'/transquest.{running}.{date}.log'
    global logger
    logger = set_logger(logger, log_file)
    logger.info('Set logger(s)')

    languages: List[str] = args['<PAIR>']

    model_type = args['--model-type'] if args['--model-type'] else MODEL_TYPE
    model_name = args['--model-name'] if args['--model-name'] else MODEL_NAME
    model_dir = args['--model-dir'] if args['--model-dir'] else None
    save_dev = not bool(args['--no-save-dev'])

    (train_df_list, dev_df_list, test_df_list, train_index_list,
     dev_index_list, test_index_list) = build_dataset(args, languages)

    # Set configs
    config = mono_config
    if model_dir is not None:
        config['output_dir'] = model_dir + '/model'
        config['best_model_dir'] = model_dir + '/best_model'
        config['tensorboard_dir'] = model_dir + '/.runs'
    if args['--cache-dir']:
        config['cache_dir'] = args['--cache-dir']
    if args['--epoch']:
        config['num_train_epochs'] = int(args['--epoch'])
    if args['--batch']:
        config['train_batch_size'] = int(args['--batch'])
    if args['--lr']:
        config['learning_rate'] = float(args['--lr'])

    logger.info('Start!')
    # Train/test model
    train_df_list, dev_df_list, test_df_list = run_monotransquest(
        config, model_type, model_name,
        train_df_list, dev_df_list, test_df_list
    )

    # Save predictions
    save_results(
        train_df_list, dev_df_list, test_df_list,
        train_index_list, dev_index_list, test_index_list, languages,
        output_dir, SUBMISSION_FILE, 'MTransQuest', save_dev, logger
    )


if __name__ == '__main__':
    main()
