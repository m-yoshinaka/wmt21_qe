#!/bin/bash

WORK_DIR=$HOME/wmt21_qe
DATA_DIR=$WORK_DIR/data
DETOK=$WORK_DIR/mosesdecoder/scripts/tokenizer/detokenizer.perl

git clone --depth 1 https://github.com/sheffieldnlp/mlqe-pe /tmp/mlqe-pe
mv /tmp/mlqe-pe/data/direct-assessments/ $DATA_DIR
mv /tmp/mlqe-pe/data/test21/ $DATA_DIR/
mv $DATA_DIR/test21/zero-shot/*.tar.gz $DATA_DIR/test21/; rm -rf $DATA_DIR/test21/zero-shot

for type in "train" "dev" "test" "test21"; do
  cd $DATA_DIR/$type
  for task in "en-de" "en-zh" "et-en" "ne-en" "ro-en" "ru-en" "si-en" "en-cs" "en-ja" "km-en" "ps-en"; do
    f=$task-$type.tar.gz
    if [ -e $f ]; then
      tar xzf $f; rm $f
      if [ -d $task-$type ]; then
        mv $task-$type $task
      fi
    fi
  done
done

# Prepare test20 data to use for evaluation script
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en"
python3 $WORK_DIR/utils/result2submit.py $tasks --prefix test20 -d $DATA_DIR/test/ -o all_lp_gold.tsv --label "z_mean" --method "Gold"

# Convert test21 data to tsv
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en en-cs en-ja km-en ps-en"
python3 $WORK_DIR/utils/convert_test21.py $tasks -d $DATA_DIR/test21/


##### Data Augmentation #####
DATA_DIR=$DATA_DIR/aug

mkdir -p $DATA_DIR
cd $DATA_DIR
for task in "en-de" "en-zh" "et-en" "ne-en" "ro-en" "si-en" ; do
  url=https://www.quest.dcs.shef.ac.uk/wmt20_files_qe/training_$task.tar.gz
  curl -LO $url
  tar xzf training_$task.tar.gz; rm training_$task.tar.gz
  _list=(${task//-/ }); src=${_list[0]}; tgt=${_list[1]}
  cat $DATA_DIR/train.${src}${tgt}.$src | $DETOK -q -l $src > $DATA_DIR/train.${src}${tgt}.detok.$src
  cat $DATA_DIR/train.${src}${tgt}.$tgt | $DETOK -q -l $tgt > $DATA_DIR/train.${src}${tgt}.detok.$tgt
done

python3 $WORK_DIR/utils/data_augmentation.py "en-de" "en-zh" "et-en" "ne-en" "ro-en" "si-en" -d $DATA_DIR/ --originals $DATA_DIR/../train/ --size 2000
mkdir -p $DATA_DIR/../train/da-da
cp $DATA_DIR/train.dada.*.tsv $DATA_DIR/../train/da-da/train.dada.df.short.tsv
