# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=/data/data/faster_rcnn_resnet101_rsna_mix.config
MODEL_DIR=/data/data/models/faster_rcnn_resnet101/single_gpu

python /tensorflow/models/research/object_detection/legacy/train.py \
          --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
          --train_dir=${MODEL_DIR} \
          --num_clones=1 --ps_tasks=1 \
	  --logtostderr
