# From the tensorflow/models/research/ directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

PIPELINE_CONFIG_PATH=/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/models/model/faster_rcnn_resnet50_rsna.config
MODEL_DIR=/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/models/model/saved_model

python ./object_detection/legacy/train.py \
       --logtostderr \
       --train_dir=${MODEL_DIR} \
       --pipeline_config_path=${PIPELINE_CONFIG_PATH}

