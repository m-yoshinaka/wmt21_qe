"""
Overview:
    Train/test TransQuest on WMT'20 dataset.
    Two runnable modules:
    `$ python3 -m wmt21_qe.tranquest ...`, or
    `$ python3 -m wmt21_qe.siamese_tranquest ...`.

Usage:
    __main__.py train (<PAIR>...) (--train <DIR>) (--dev <DIR>) [--test <DIR>]
                      [--model-type <MODEL>] [--model-name <MODEL>] [--model-dir <DIR>]
                      [--epoch <EPOCH>] [--batch <BATCH>] [--lr <LR>]
                      (--out-dir <DIR>) [--cache-dir <DIR>]
    __main__.py eval  (<PAIR>...) (--dev <DIR>) [--test <DIR>] [--no-save-dev]
                      [--model-type <MODEL>] [--model-name <MODEL>] [--model-dir <DIR>]
                      (--out-dir <DIR>) [--cache-dir <DIR>]
    __main__.py -h|--help

Options:
    -h,--help                   Show this help message
    train|eval                  Running mode option
    <PAIR>...                   Language pair(s) sequence, e.g. 'en-de en-zh ... si-en'
    --train <DIR>               Directory containing tsv files of training data
    --dev <DIR>                 Directory containing tsv files of dev data
    --test <DIR>                Directory containing tsv files of test data
    --model-type <MODEL>        Model type (optional, by default 'xlmroberta')
    --model-name <MODEL>        Model name (optional, by default 'TransQuest/monotransquest-da-multilingual')
    --model-dir <DIR>           Model directory (optional, by default '.temp/outputs/')
    --epoch <EPOCH>             # of epochs (optional, by default 10)
    --batch <BATCH>             Batch size (optional, by default 8)
    --lr <LR>                   Learning rate (optional, by default 2e-5)
    --out-dir <DIR>             Directory to save results
    --no-save-dev               Not to save results on dev dat (optional, by default False)
    --cache-dir <DIR>           Argument (optional, by default '.cache/')
"""  # noqa:E501


def main():
    print(__doc__)


if __name__ == '__main__':
    main()
