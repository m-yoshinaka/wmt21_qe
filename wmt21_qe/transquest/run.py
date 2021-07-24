from typing import List, Tuple
from logging import getLogger
from pathlib import Path
from pprint import pformat

import pandas as pd
from torch import cuda, manual_seed
from sklearn.metrics import mean_absolute_error

from TransQuest.examples.sentence_level.wmt_2020.common.util.normalizer import fit, un_fit  # noqa: E501
from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr  # noqa: E501
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel  # noqa: E501

from wmt21_qe.transquest.config import MODEL_NAME, MODEL_TYPE, SEED


manual_seed(SEED)
logger = getLogger('WMT21QE')


class Trainer(object):

    def __init__(
        self, model_type: str, model_name: str, config: dict = None
    ) -> None:
        self.gpu = cuda.is_available()
        self.config = config
        self.model_type = model_type
        self.model_name = model_name
        self.model_dir = self.config['output_dir']
        self.model = MonoTransQuestModel(
            self.model_type, self.model_name, num_labels=1, use_cuda=self.gpu,
            args=self.config
        )
        self._reloaded = False

    def _reload_model(self) -> None:
        if self.model_dir is not None \
                and Path(self.config['best_model_dir']).exists() \
                and not self._reloaded:
            logger.info('Reloading best checkpoint ...')
            self.model = MonoTransQuestModel(
                self.model_type, self.config['best_model_dir'], num_labels=1,
                use_cuda=self.gpu, args=self.config
            )
        self._reloaded = True

    def fit(self, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
        logger.info('Start model training')
        global_step, details = self.model.train_model(
            train_df=train_df, eval_df=eval_df, output_dir=self.model_dir,
            args=self.config
        )
        logger.info(f'Global Steps: {global_step}')
        logger.info('Details: \n{}\n{}'.format(
            list(details.keys()), pformat(list(zip(*details.values())))
        ))

    def eval(self, eval_df: pd.DataFrame) -> list:
        self._reload_model()
        logger.info('Evaluating validation data ...')
        result, model_outputs, _ = self.model.eval_model(
            eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
            mae=mean_absolute_error
        )
        logger.info(f'Results: {result}')
        return model_outputs

    def predict(self, test_df: pd.DataFrame) -> list:
        self._reload_model()
        sentence_pairs = list(map(list, zip(test_df['text_a'].to_list(),
                                            test_df['text_b'].to_list())))
        predictions, _ = self.model.predict(sentence_pairs)
        return predictions


def run_monotransquest(
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
            train_model_outputs = trainer.eval(_train_df)
            dev_model_outputs = trainer.eval(_dev_df)
            _train_df['predictions'] = train_model_outputs
            _dev_df['predictions'] = dev_model_outputs
            # Normalize
            _train_df = un_fit(_train_df, 'predictions')
            _dev_df = un_fit(_dev_df, 'predictions')
            # Append results on a language pair
            new_train_df_list.append(_train_df)
            new_dev_df_list.append(_dev_df)

    elif dev_df is not None:
        for _dev_df in dev_df_list:
            dev_model_outputs = trainer.eval(_dev_df)
            _dev_df['predictions'] = dev_model_outputs
            _dev_df = un_fit(_dev_df, 'predictions')
            new_dev_df_list.append(_dev_df)

    if test_df_list is not None:
        for _test_df in test_df_list:
            predictions = trainer.predict(_test_df)
            _test_df['predictions'] = predictions
            _test_df = un_fit(_test_df, 'predictions')
            new_test_df_list.append(_test_df)

    return new_train_df_list, new_dev_df_list, new_test_df_list
