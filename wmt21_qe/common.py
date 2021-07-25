import os
from typing import List, Optional, Tuple
from logging import Logger
from sklearn import preprocessing

from wmt21_qe.utils import (
    find_data_file, read_annotated_file, read_test_file, format_submission
)


min_max_scaler = preprocessing.MinMaxScaler()


def fit(df, label):
    x = df[[label]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x)
    df[label] = x_scaled
    return df


def un_fit(df, label):
    x = df[[label]].values.astype(float)
    x_unscaled = min_max_scaler.inverse_transform(x)
    df[label] = x_unscaled
    return df


def build_dataset(
    args: dict, languages: List[str]
) -> Tuple[list]:
    train_df_list = []
    dev_df_list = []
    test_df_list = []
    train_index_list = []
    dev_index_list = []
    test_index_list = []

    for lang_pair in languages:
        src, tgt = lang_pair.split('-')

        _index = 'segid' if lang_pair == 'ru-en' else 'index'
        _column_mv = {'original': 'text_a', 'translation': 'text_b',
                      'z_mean': 'labels'}
        _test_column_mv = {'original': 'text_a', 'translation': 'text_b'}

        if args['--train']:
            train_f = find_data_file(args['--train'], src, tgt)
            if train_f is not None:
                train_df = read_annotated_file(train_f, _index)
                train_df = train_df[
                    ['index', 'original', 'translation', 'z_mean']
                ]
                train_df = train_df.rename(columns=_column_mv).dropna()
                train_df_list.append(train_df)
                train_index_list.append(train_df['index'].to_list())

        dev_f = find_data_file(args['--dev'], src, tgt)
        if dev_f is not None:
            dev_df = read_annotated_file(dev_f, _index)
            dev_df = dev_df[
                ['index', 'original', 'translation', 'z_mean']
            ]
            dev_df = dev_df.rename(columns=_column_mv).dropna()
            dev_df_list.append(dev_df)
            dev_index_list.append(dev_df['index'].to_list())

        if args['--test']:
            test_f = find_data_file(args['--test'], src, tgt)
            if test_f is not None:
                test_df = read_test_file(test_f, _index)
                try:
                    test_df = test_df[[
                        ['index', 'original', 'translation', 'z_mean']
                    ]]
                    test_df = test_df.rename(columns=_column_mv).dropna()
                except KeyError:
                    test_df = test_df[['index', 'original', 'translation']]
                    test_df = test_df.rename(columns=_test_column_mv).dropna()
                test_df_list.append(test_df)
                test_index_list.append(test_df['index'].to_list())

    train_df_list = train_df_list if train_df_list else None
    test_df_list = test_df_list if test_df_list else None
    test_index_list = test_index_list if test_index_list else None

    return (train_df_list, dev_df_list, test_df_list,
            train_index_list, dev_index_list, test_index_list)


def save_results(
    train_df_list: Optional[list],
    dev_df_list: Optional[list],
    test_df_list: Optional[list],
    train_index_list: Optional[list],
    dev_index_list: Optional[list],
    test_index_list: Optional[list],
    languages: List[str],
    output_dir: str,
    submit_f: str,
    method: str,
    save_dev: bool,
    logger: Logger
) -> None:

    if train_df_list:
        fp = open(os.path.join(output_dir, f'train.{submit_f}'), 'w')
        for df, idx, lp in zip(train_df_list, train_index_list, languages):
            logger.info(f'Saving results on {lp} of train set ...')
            format_submission(df, lp, method, index=idx, fp=fp)
        fp.close()

    if save_dev:
        fp = open(os.path.join(output_dir, f'dev.{submit_f}'), 'w')
        for df, idx, lp in zip(dev_df_list, dev_index_list, languages):
            logger.info(f'Saving results on {lp} of dev set ...')
            format_submission(df, lp, method, index=idx, fp=fp)
        fp.close()

    if test_df_list and test_index_list is not None:
        fp = open(os.path.join(output_dir, f'test.{submit_f}'), 'w')
        for df, idx, lp in zip(test_df_list, test_index_list, languages):
            logger.info(f'Saving results on {lp} of test set ...')
            format_submission(df, lp, method, index=idx, fp=fp)
        fp.close()
