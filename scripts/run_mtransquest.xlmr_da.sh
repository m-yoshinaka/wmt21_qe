#!/bin/bash

WORK_DIR=$HOME/wmt21_qe
cd $WORK_DIR

if [ -z "$1" ]; then
  echo "Usage: ./run_mtransquest.xlmr.sh CUDA_VISIBLE_DEVICES"; exit
fi

CACHE_DIR=$WORK_DIR/.cache
DATA_DIR=$WORK_DIR/data

###############################################################################
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en en-cs en-ja km-en ps-en da-da"
###############################################################################

OUT_DIR=$WORK_DIR/results/mtransquest.xlmr_da_1
MODEL_DIR=$WORK_DIR/models/mtransquest.xlmr_da_1

# Train on train20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test/ --model-dir $MODEL_DIR --epoch 3 --lr 2e-5 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.30828700763150757
#          for en-zh...
#         pearson: 0.41025311011296234
#          for et-en...
#         pearson: 0.7213368070526384
#          for ne-en...
#         pearson: 0.7310928079491567
#          for ro-en...
#         pearson: 0.8598612842423796
#          for ru-en...
#         pearson: 0.735680452696897
#          for si-en...
#         pearson: 0.615966976609712
#          averaging...
# done.
# pearson: 0.6260683494707505
# mae: 0.5090281644499893
# rmse: 0.6387056076292986

# Computing scores...
#          for en-de...
#         pearson: 0.4193007882956748
#          for en-zh...
#         pearson: 0.41165194759142115
#          for et-en...
#         pearson: 0.7454709728610056
#          for ne-en...
#         pearson: 0.7372302781693361
#          for ro-en...
#         pearson: 0.8057546724233863
#          for ru-en...
#         pearson: 0.7353563474813596
#          for si-en...
#         pearson: 0.6148856459208747
#          averaging...
# done.
# pearson: 0.6385215218204369
# mae: 0.5595494566177345
# rmse: 0.7374116032696859
