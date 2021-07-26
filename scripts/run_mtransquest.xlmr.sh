#!/bin/bash

WORK_DIR=$HOME/wmt21_qe
cd $WORK_DIR

if [ -z "$1" ]; then
  echo "Usage: ./run_mtransquest.xlmr.sh CUDA_VISIBLE_DEVICES"; exit
fi

CACHE_DIR=$WORK_DIR/.cache
DATA_DIR=$WORK_DIR/data

#########################################################################
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en en-cs en-ja km-en ps-en"
#########################################################################

OUT_DIR=$WORK_DIR/results/mtransquest.xlmr
MODEL_DIR=$WORK_DIR/models/mtransquest.xlmr

# Train on train20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test/ --model-dir $MODEL_DIR --epoch 5 --lr 2e-5 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.42382700935085865
#          for en-zh...
#         pearson: 0.406738678027414
#          for et-en...
#         pearson: 0.7138103657501873
#          for ne-en...
#         pearson: 0.6874667529390825
#          for ro-en...
#         pearson: 0.8544616835231237
#          for ru-en...
#         pearson: 0.7286267640112433
#          for si-en...
#         pearson: 0.5834764036526935
#          averaging...
# done.
# pearson: 0.6283439510363719
# mae: 0.5136726903680874
# rmse: 0.6629269737955364
