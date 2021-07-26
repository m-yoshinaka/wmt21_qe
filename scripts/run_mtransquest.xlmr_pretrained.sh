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

OUT_DIR=$WORK_DIR/results/mtransquest.xlmr_pretrained
MODEL_DIR=$WORK_DIR/models/mtransquest.xlmr_pretrained

# Infer on train20/dev20/test20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test/ --model-dir $MODEL_DIR --epoch 0 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.4241757865904014
#          for en-zh...
#         pearson: 0.441813530746759
#          for et-en...
#         pearson: 0.7657842161741957
#          for ne-en...
#         pearson: 0.754754783089823
#          for ro-en...
#         pearson: 0.8797114071255278
#          for ru-en...
#         pearson: 0.7584050764944787
#          for si-en...
#         pearson: 0.5885826447400659
#          averaging...
# done.
# pearson: 0.6590324921373217
# mae: 0.9260557542148538
# rmse: 1.0896527599051835
