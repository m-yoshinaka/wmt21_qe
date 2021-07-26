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


###############################################################################
tasks="en-de en-zh et-en ne-en ro-en ru-en si-en en-cs en-ja km-en ps-en da-da"
###############################################################################

OUT_DIR=$WORK_DIR/results/mtransquest.labse_da_1
MODEL_DIR=$WORK_DIR/models/mtransquest.labse_da_1

# Train on train20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test/ --model-type "bert" --model-name $MODEL --model-dir $MODEL_DIR --epoch 3 --lr 2e-5 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-type "bert" --model-name $MODEL --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.44221634300048573
#          for en-zh...
#         pearson: 0.40000465050667416
#          for et-en...
#         pearson: 0.7483927566220884
#          for ne-en...
#         pearson: 0.7605046456900161
#          for ro-en...
#         pearson: 0.8628277523874226
#          for ru-en...
#         pearson: 0.7265456896876054
#          for si-en...
#         pearson: 0.63558660515578
#          averaging...
# done.
# pearson: 0.653725491864296
# mae: 0.5141718009013224
# rmse: 0.6774554785358765

# Computing scores...
#          for en-de...
#         pearson: 0.47131585240514506
#          for en-zh...
#         pearson: 0.4056867087592656
#          for et-en...
#         pearson: 0.7490789152723792
#          for ne-en...
#         pearson: 0.7633543191310276
#          for ro-en...
#         pearson: 0.8427522441812325
#          for ru-en...
#         pearson: 0.7309633138379675
#          for si-en...
#         pearson: 0.6388492358210501
#          averaging...
# done.
# pearson: 0.657428655629724
# mae: 0.5080809534936871
# rmse: 0.6733551160494874
