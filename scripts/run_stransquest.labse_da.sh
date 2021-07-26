#!/bin/bash

WORK_DIR=$HOME/wmt21_qe
cd $WORK_DIR

if [ -z "$1" ]; then
  echo "Usage: ./run_stransquest.labse.sh CUDA_VISIBLE_DEVICES"; exit
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

OUT_DIR=$WORK_DIR/results/stransquest.labse_da_1
MODEL_DIR=$WORK_DIR/models/stransquest.labse_da_1

# Infer on train20
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.siamese_transquest train $tasks --train $DATA_DIR/train/ --dev $DATA_DIR/dev/ --test $DATA_DIR/test/ --model-type "bert" --model-name $MODEL --model-dir $MODEL_DIR --epoch 3 --lr 2e-5 --out-dir $OUT_DIR --cache-dir $CACHE_DIR

# Infer on test21
CUDA_VISIBLE_DEVICES=$1 python3 -m wmt21_qe.siamese_transquest eval $tasks --dev $DATA_DIR/test/ --test $DATA_DIR/test21/ --no-save-dev --model-type "bert" --model-name $MODEL --model-dir $MODEL_DIR --out-dir $OUT_DIR/test21 --cache-dir $CACHE_DIR

# Evaluate on test20
python3 $WORK_DIR/utils/evaluate.py $OUT_DIR/test.predictions.txt $DATA_DIR/test/all_lp_gold.tsv

# Computing scores...
#          for en-de...
#         pearson: 0.018120247850233567
#          for en-zh...
#         pearson: 0.32889903535116005
#          for et-en...
#         pearson: 0.6973960436805181
#          for ne-en...
#         pearson: 0.6606211672541927
#          for ro-en...
#         pearson: 0.8365921133181843
#          for ru-en...
#         pearson: 0.6827152124274684
#          for si-en...
#         pearson: 0.5672215708896636
#          averaging...
# done.
# pearson: 0.5416521986816315
# mae: 0.6454328343441047
# rmse: 0.8391505536461924

# Computing scores...
#          for en-de...
#         pearson: 0.04708850975025189
#          for en-zh...
#         pearson: 0.33638724330314457
#          for et-en...
#         pearson: 0.6753136417071289
#          for ne-en...
#         pearson: 0.6450794013983393
#          for ro-en...
#         pearson: 0.8285663450004045
#          for ru-en...
#         pearson: 0.700692119643135
#          for si-en...
#         pearson: 0.54147487298276
#          averaging...
# done.
# pearson: 0.5392288762550235
# mae: 0.6846221446386133
# rmse: 0.8793342309237462
