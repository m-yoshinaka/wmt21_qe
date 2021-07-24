import csv
from typing import TextIO, Optional
from pathlib import Path
from logging import INFO, Formatter, StreamHandler, FileHandler, Logger

import pandas as pd
from sklearn.metrics import mean_absolute_error

from transquest.algo.sentence_level.monotransquest.evaluation import (
    pearson_corr, spearman_corr, rmse
)


def set_logger(logger: Logger, log_file: Optional[str] = None) -> Logger:
    logger.setLevel(INFO)
    formatter = Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    if log_file is not None:
        f_handler = FileHandler(log_file)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    s_handler = StreamHandler()
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)
    return logger


def find_data_file(base_dir: str, src: str, tgt: str) -> Optional[str]:
    try:
        data_dir = Path(base_dir + f'/{src}-{tgt}/')
        assert data_dir.exists(), FileNotFoundError(f'{data_dir}')
        data_f = next(iter(data_dir.glob('*.df.short.tsv')))
        return str(data_f)
    except AssertionError:
        return None


def read_annotated_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    z_means = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t",
                                quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            z_means.append(float(row["z_mean"]))

    return pd.DataFrame(
        {'index': indices, 'original': originals, 'translation': translations,
         'z_mean': z_means}
    )


def read_test_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t",
                                quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])

    return pd.DataFrame(
        {'index': indices, 'original': originals, 'translation': translations}
    )


def stat_text(data_frame, label_column, prediction_column) -> str:
    data_frame = data_frame.sort_values(label_column)

    ground_trues = data_frame[label_column].tolist()
    predictions = data_frame[prediction_column].tolist()
    pearson = pearson_corr(ground_trues, predictions)
    spearman = spearman_corr(ground_trues, predictions)
    rmse_value = rmse(ground_trues, predictions)
    mae = mean_absolute_error(ground_trues, predictions)

    textstr = f'RMSE={rmse_value:.4f}\n'\
              f'MAE={mae:.4f}\n'\
              f'Pearson Correlation={pearson:.4f}\n'\
              f'Spearman Correlation={spearman:.4f}'
    return textstr


def format_submission(
    df: pd.DataFrame, language_pair: str, method: str, index: list, fp: TextIO,
    index_type=None
):
    if index_type is None:
        index = index
    elif index_type == "Auto":
        index = range(0, df.shape[0])

    predictions = df['predictions']
    for number, prediction in zip(index, predictions):
        text = language_pair + "\t" + method + "\t" + \
            str(number) + "\t" + str(prediction)
        fp.write("%s\n" % text)
