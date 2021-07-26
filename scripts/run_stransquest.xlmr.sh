#!/bin/bash

WORK_DIR=$HOME/wmt21_qe
cd $WORK_DIR

if [ -z "$1" ]; then
  echo "Usage: ./run_stransquest.xlmr.sh CUDA_VISIBLE_DEVICES"; exit
fi

CACHE_DIR=$WORK_DIR/.cache
DATA_DIR=$WORK_DIR/data

#########################################################################
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en en-cs en-ja km-en ps-en"
#########################################################################

OUT_DIR=$WORK_DIR/results/stransquest.xlmr
MODEL_DIR=$WORK_DIR/models/stransquest.xlmr

# Train on train20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.siamese_transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test --model-name "sentence-transformers/paraphrase-xlm-r-multilingual-v1" --model-dir $MODEL_DIR --epoch 3 --lr 2e-5 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.siamese_transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-name "sentence-transformers/paraphrase-xlm-r-multilingual-v1" --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.1435714463728557
#          for en-zh...
#         pearson: 0.31062952974552777
#          for et-en...
#         pearson: 0.7114879650338461
#          for ne-en...
#         pearson: 0.6961008488021242
#          for ro-en...
#         pearson: 0.823860089787878
#          for ru-en...
#         pearson: 0.6753071393580585
#          for si-en...
#         pearson: 0.5638990740015187
#          averaging...
# done.
# pearson: 0.5606937275859727
# mae: 0.6146526919147007
# rmse: 0.8006512763008745
