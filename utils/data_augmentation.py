"""
Usage:
    script.py [-h|--help] (<PAIR>...) (-d <DIR>) (--originals <DIR>) (--size <N>)

Options:
    -h,--help           Show this help message
    <PAIR>...           Language pair(s) sequence, e.g. 'en-de en-zh ... si-en'
    -d <DIR>            Directory containing WMT'21 training data of MT system
    --originals <DIR>   Directory containing original training files
    --size <N>          # of data to sample
"""  # noqa:E501

import csv
from logging import getLogger, INFO
from typing import List
from pathlib import Path
from docopt import docopt
import pandas as pd

from statistics import mean
from sklearn.preprocessing import StandardScaler


logger = getLogger('DataAugmentation')
logger.setLevel(INFO)


def read_annotated_file(path, index='index'):
    indices = []
    originals = []
    translations = []
    scores = []
    z_means = []
    with open(path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t',
                                quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row['original'])
            translations.append(row['translation'])
            scores.append(row['scores'])
            z_means.append(float(row['z_mean']))

    return pd.DataFrame(
        {'index': indices, 'original': originals, 'translation': translations,
         'scores': scores, 'z_mean': z_means}
    )


def build_dataframe(
    data_dir: Path, orig_dir: Path, lp: str, index: str = 'index'
) -> pd.DataFrame:
    s, t = lp.split('-')

    orig_f = orig_dir / f'{s}-{t}/train.{s}{t}.df.short.tsv'
    df = read_annotated_file(orig_f, index)

    max_scores = []
    a_list, b_list, c_list = [], [], []
    for item in df['scores']:
        item = tuple(map(int, item[1:-1].split(', ')))
        a_list.append(item[0])
        b_list.append(item[1])
        c_list.append(item[2])
    df['a_scores'] = a_list
    df['b_scores'] = b_list
    df['c_scores'] = c_list
    for col in ['a_scores', 'b_scores', 'c_scores']:
        scaler = StandardScaler()
        scaler.fit(df[[col]])
        score = scaler.transform([[100]])
        max_scores.append(score[0][0])
        del scaler
    max_score_mean = mean(max_scores)

    logger.info(' - loading MT training data ...')
    src_f = data_dir / f'train.{s}{t}.detok.{s}'
    hyp_f = data_dir / f'train.{s}{t}.detok.{t}'
    src_fp = src_f.open('r')
    hyp_fp = hyp_f.open('r')
    data = []
    for s, hyp in zip(src_fp, hyp_fp):
        s = s.strip()
        hyp = hyp.strip()
        if s and hyp:
            data.append([s.strip(), hyp.strip(), max_score_mean])
    src_fp.close()
    hyp_fp.close()

    df = pd.DataFrame(data=data, columns=['original', 'translation', 'z_mean'])
    df.index.name = index
    return df


def main():
    args = docopt(__doc__)

    language_pairs: List[str] = args['<PAIR>']
    data_dir = Path(args['-d'])
    orig_dir = Path(args['--originals'])
    size = int(args['--size'])

    df_list = []
    for i, lp in enumerate(language_pairs):
        logger.info(f'Processing {lp} ...')
        index = 'segid' if lp == 'ru-en' else 'index'
        df = build_dataframe(data_dir, orig_dir, lp, index=index)
        df = df.sample(n=size, random_state=42 * i)
        df_list.append(df)

    df = pd.concat(df_list)
    # df = df.sample(n=size, random_state=42)
    df = df.reset_index(drop=True)
    df.index.name = 'index'
    logger.info(f'\n{df}')

    logger.info('Saving results ...')
    out_f = data_dir / f'train.dada.df.{size}.tsv'
    df.to_csv(str(out_f), header=True, index=True, sep='\t',
              encoding='utf-8', quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    main()
