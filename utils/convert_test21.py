"""
Usage:
    script.py [-h|--help] (<PAIR>...) (-d <DIR>)

Options:
    -h,--help       Show this help message
    <PAIR>...       Language pair(s) sequence, e.g. 'en-de en-zh ... si-en'
    -d <DIR>        Base directory containing WMT'21 test data
"""

import csv
from typing import List
from pathlib import Path
from docopt import docopt
import pandas as pd


def build_dataframe(data_dir: Path, index: str = 'index') -> pd.DataFrame:
    src_f = data_dir / 'test21.src'
    hyp_f = data_dir / 'test21.mt'
    src_fp = src_f.open('r')
    hyp_fp = hyp_f.open('r')

    data = []
    for src, hyp in zip(src_fp, hyp_fp):
        src = src.strip()
        hyp = hyp.strip()
        if src and hyp:
            data.append([src.strip(), hyp.strip()])
    src_fp.close()
    hyp_fp.close()

    df = pd.DataFrame(data=data, columns=['original', 'translation'])
    df.index.name = index
    return df


# index original translation
def main():
    args = docopt(__doc__)

    language_pairs: List[str] = args['<PAIR>']
    base_dir = Path(args['-d'])

    for lp in language_pairs:
        data_dir = base_dir / lp
        index = 'segid' if lp == 'ru-en' else 'index'
        df = build_dataframe(data_dir, index=index)

        out_f = data_dir / 'test21.{}.df.short.tsv'.format(lp.replace('-', ''))
        df.to_csv(str(out_f), header=True, index=True, sep='\t',
                  encoding='utf-8-sig', quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    main()
