# From the tensorflow/models/research/ directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#CUDA_VISIBLE_DEVICES=1  #not working!
PIPELINE_CONFIG_PATH=/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/models/model/faster_rcnn_resnet50_rsna.config
MODEL_DIR=/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/models/model/saved_model
NUM_TRAIN_STEPS=50000
NUM_EVAL_STEPS=2000
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr \
    --gpudev=1  # added a new argument, remove this argument if only one GPU
