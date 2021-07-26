"""
Usage:
    result2submit.py [-h|--help] (<PAIR>...) (--prefix <TEXT>)
                     (-d <DIR>) (-o <FILE>) [--overwrite]
                     [--label <LABEL>] [--method <NAME>]

Options:
    -h,--help       Show this help message
    <PAIR>...       Language pair(s) sequence, e.g. 'en-de en-zh ... si-en'
    --prefix <TEXT> Tsv file prefix: name of data subset, e.g. train, dev, test
    -d <DIR>        Directory containing tsv file(s)
    -o <FILE>       Submission file *NAME*
    --overwrite     Overwrite output file
    --label <LABEL> Column to read (by default 'predictions')
    --method <NAME> Method name (optional, by default 'TransQuest')

"""  # noqa:E501

from typing import List, TextIO
from pathlib import Path
from docopt import docopt
import pandas as pd


def find_data_file(data_dir: Path, prefix: str, src: str, tgt: str) -> Path:
    assert data_dir.exists(), FileNotFoundError(f'{data_dir}')
    try:
        data_f = next(iter(data_dir.glob(f'**/{prefix}.{src}*{tgt}*.tsv')))
        return data_f
    except StopIteration:
        return None


def format_submission(
    df: pd.DataFrame, language_pair: str, method: str, index: list, fp: TextIO,
    index_type=None, label: str = 'predictions'
):
    if index_type is None:
        index = index
    elif index_type == "Auto":
        index = range(0, df.shape[0])

    predictions = df[label]
    for number, prediction in zip(index, predictions):
        text = language_pair + "\t" + method + "\t" + \
            str(number) + "\t" + str(prediction)
        fp.write("%s\n" % text)


def main():
    args = docopt(__doc__)

    languages: List[str] = args['<PAIR>']
    prefix: str = args['--prefix']
    data_dir = Path(args['-d'])
    out_file = data_dir / str(args['-o'])
    overwrite = bool(args['--overwrite'])

    label: str = args['--label'] if args['--label'] else 'predictions'
    method: str = args['--method'] if args['--method'] else 'TransQuest'

    print(languages)
    assert not out_file.exists() or overwrite, 'File exits: stop overwriting.'

    fp = out_file.open('w')
    for lp in languages:
        src, tgt = lp.split('-')
        data_f = find_data_file(data_dir, prefix, src, tgt)
        if data_f is None:
            print(f'Diffrent format: {lp}')
        df = pd.read_csv(data_f, sep='\t')
        index_name = 'index' if lp != 'ru-en' else 'segid'
        index = df[index_name]
        format_submission(df, lp, method=method, index=index, fp=fp,
                          label=label)
    fp.close()


if __name__ == '__main__':
    main()
