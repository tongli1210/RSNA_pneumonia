# From tensorflow/models/research/oid
# OUTPUT_DIR=/data/data/tfrecord_opacity  # or test
OUTPUT_DIR=/data/data/tfrecord_all  # or test
# TF_RECORD_FILES=/data/data/tfrecord_opacity/opacity_val.tfrecord
TF_RECORD_FILES=/data/data/tfrecord_all/opacity_val.tfrecord
MODEL_PATH=/data/data/models/faster_rcnn_resnet101/multiple_gpu
echo ${TF_RECORD_FILES}


python -m object_detection/inference/infer_detections \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=${OUTPUT_DIR}/val_detections_frcnn101_mGPU.tfrecord \
  --inference_graph=${MODEL_PATH}/frozen_inference_graph.pb \
  --discard_image_pixels
