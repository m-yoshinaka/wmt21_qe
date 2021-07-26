#!/bin/bash

WORK_DIR=$HOME/wmt21_qe
cd $WORK_DIR

if [ -z "$1" ]; then
  echo "Usage: ./run_mtransquest.labse.sh CUDA_VISIBLE_DEVICES"; exit
fi

CACHE_DIR=$WORK_DIR/.cache
DATA_DIR=$WORK_DIR/data
MODEL=$WORK_DIR/models/labse

# Convert TF model to Hugging face model
# source $HOME/.venv/tf/bin/activate
# mkdir -p $MODEL
# transformers-cli convert --model_type bert --tf_checkpoint $MODEL.tf/bert_model.ckpt --config $MODEL.tf/bert_config.json --pytorch_dump_output $MODEL/pytorch_model.bin
# cp $MODEL.tf/bert_config.json $MODEL/config.json
# cp $MODEL.tf/vocab.txt $MODEL/
# deactivate


#########################################################################
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en en-cs en-ja km-en ps-en"
#########################################################################

OUT_DIR=$WORK_DIR/results/mtransquest.labse
MODEL_DIR=$WORK_DIR/models/mtransquest.labse

# Train on train20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test/ --model-type "bert" --model-name $MODEL --model-dir $MODEL_DIR --epoch 5 --lr 2e-5 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-type "bert" --model-name $MODEL --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.4686498904170028
#          for en-zh...
#         pearson: 0.4074744580088641
#          for et-en...
#         pearson: 0.7412181110177594
#          for ne-en...
#         pearson: 0.7404345638215541
#          for ro-en...
#         pearson: 0.8461567128619116
#          for ru-en...
#         pearson: 0.696895021377152
#          for si-en...
#         pearson: 0.6043782969661232
#          averaging...
# done.
# pearson: 0.643601007781481
# mae: 0.5036988382527258
# rmse: 0.6487862594910968
