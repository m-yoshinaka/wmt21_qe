#!/bin/bash

WORK_DIR=$HOME/wmt21_qe
cd $WORK_DIR

if [ -z "$1" ]; then
  echo "Usage: ./run_stransquest.xlmr.sh CUDA_VISIBLE_DEVICES"; exit
fi

CACHE_DIR=$WORK_DIR/.cache
DATA_DIR=$WORK_DIR/data

###############################################################################
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en en-cs en-ja km-en ps-en da-da"
###############################################################################

OUT_DIR=$WORK_DIR/results/stransquest.xlmr_da_1
MODEL_DIR=$WORK_DIR/models/stransquest.xlmr_da_1

# Train on train20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.siamese_transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test --model-name "sentence-transformers/paraphrase-xlm-r-multilingual-v1" --model-dir $MODEL_DIR --epoch 3 --lr 2e-5 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.siamese_transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-name "sentence-transformers/paraphrase-xlm-r-multilingual-v1" --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.13083453388053226
#          for en-zh...
#         pearson: 0.3128272165034736
#          for et-en...
#         pearson: 0.7163945956238668
#          for ne-en...
#         pearson: 0.6883590644985343
#          for ro-en...
#         pearson: 0.8272501609335235
#          for ru-en...
#         pearson: 0.7001657674320404
#          for si-en...
#         pearson: 0.5608847301120526
#          averaging...
# done.
# pearson: 0.5623880098548605
# mae: 0.6229087782758842
# rmse: 0.8128862146723049

# Computing scores...
#          for en-de...
#         pearson: 0.010236574959785964
#          for en-zh...
#         pearson: 0.31615100096055915
#          for et-en...
#         pearson: 0.7149218488698855
#          for ne-en...
#         pearson: 0.7064087416815032
#          for ro-en...
#         pearson: 0.802685440299531
#          for ru-en...
#         pearson: 0.695335899428484
#          for si-en...
#         pearson: 0.5813228356242172
#          averaging...
# done.
# pearson: 0.5467231916891381
# mae: 0.7389207268230015
# rmse: 0.9337226240446607
