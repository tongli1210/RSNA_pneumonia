# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/models/model/faster_rcnn_resnet50_pets.config
MODEL_DIR=/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/models/model
NUM_TRAIN_STEPS=50000
NUM_EVAL_STEPS=2000
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr