from typing import List, Tuple
from logging import getLogger
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from torch import manual_seed

from transquest.algo.sentence_level.siamesetransquest import SiameseTransQuestModel  # noqa:E501

from wmt21_qe.common import fit, un_fit
from wmt21_qe.utils import stat_text
from wmt21_qe.siamese_transquest.config import MODEL_NAME, MODEL_TYPE, SEED


manual_seed(SEED)
logger = getLogger('WMT21QE')


class Trainer(object):

    def __init__(
        self, model_type: str, model_name: str, config: dict = None
    ) -> None:
        self.config = config
        self.model_type = model_type
        self.model_name = model_name
        self.model_dir = self.config['output_dir']
        self.model = SiameseTransQuestModel(model_name, args=self.config)
        self._reloaded = False

    def _reload_model(self) -> None:
        if self.model_dir is not None \
                and Path(self.config['best_model_dir']).exists() \
                and not self._reloaded:
            logger.info('Reloading best checkpoint ...')
            self.model = SiameseTransQuestModel(
                self.config['best_model_dir'], args=self.config
            )
        self._reloaded = True

    def fit(self, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
        logger.info('Start model training')
        self.model.train_model(
            train_df=train_df, eval_df=eval_df, output_dir=self.model_dir,
            args=self.config
        )

    def predict(self, eval_df: pd.DataFrame) -> list:
        self._reload_model()
        sentence_pairs = list(map(list, zip(eval_df['text_a'].to_list(),
                                            eval_df['text_b'].to_list())))
        model_outs = self.model.predict(sentence_pairs)
        return model_outs.tolist() if isinstance(model_outs, np.ndarray) \
            else model_outs


def run_siamesetransquest(
    config: dict,
    model_type: str = MODEL_TYPE,
    model_name: str = MODEL_NAME,
    train_df_list: List[pd.DataFrame] = None,
    dev_df_list: List[pd.DataFrame] = None,
    test_df_list: List[pd.DataFrame] = None,
) -> Tuple[list, list, list]:

    new_train_df_list = []
    new_dev_df_list = []
    new_test_df_list = []

    trainer = Trainer(model_type, model_name, config)

    train_df = pd.concat(train_df_list) if train_df_list is not None else None
    dev_df = pd.concat(dev_df_list) if dev_df_list is not None else None
    train_df = fit(train_df, 'labels') if train_df is not None else None
    dev_df = fit(dev_df, 'labels') if dev_df is not None else None

    if train_df is not None and dev_df is not None:
        logger.info(f'Configs:\n{pformat(config)}')
        logger.info(f'Train data:\n{train_df}')
        # Training
        trainer.fit(train_df, dev_df)
        # Validation
        for _train_df, _dev_df in zip(train_df_list, dev_df_list):
            logger.info('Evaluating validation data ...')
            train_model_outputs = trainer.predict(_train_df)
            dev_model_outputs = trainer.predict(_dev_df)
            _train_df['predictions'] = train_model_outputs
            _dev_df['predictions'] = dev_model_outputs
            # Normalize
            _train_df = un_fit(_train_df, 'predictions')
            _dev_df = un_fit(_dev_df, 'predictions')
            # Append results on a language pair
            new_train_df_list.append(_train_df)
            new_dev_df_list.append(_dev_df)

            results_str = stat_text(
                _dev_df, 'labels', 'predictions'
            ).replace('\n', ', ')
            logger.info(f'Results:  {results_str}')

    elif dev_df is not None:
        for _dev_df in dev_df_list:
            logger.info('Evaluating validation data ...')

            dev_model_outputs = trainer.predict(_dev_df)
            _dev_df['predictions'] = dev_model_outputs
            _dev_df = un_fit(_dev_df, 'predictions')
            new_dev_df_list.append(_dev_df)

            results_str = stat_text(
                _dev_df, 'labels', 'predictions'
            ).replace('\n', ', ')
            logger.info(f'Results:  {results_str}')

    if test_df_list is not None:
        for _test_df in test_df_list:
            predictions = trainer.predict(_test_df)
            _test_df['predictions'] = predictions
            _test_df = un_fit(_test_df, 'predictions')
            new_test_df_list.append(_test_df)

    return new_train_df_list, new_dev_df_list, new_test_df_list
