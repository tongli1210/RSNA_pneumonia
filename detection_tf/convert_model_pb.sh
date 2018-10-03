python /tensorflow/models/research/object_detection/export_inference_graph.py \
	--input_type image_tensor \
	--pipeline_config_path /data/data/faster_rcnn_resnet101_mGPU_rsna.config \
	--trained_checkpoint_prefix /data/data/models/faster_rcnn_resnet101/multiple_gpu/model.ckpt-100000 \
	--output_directory /data/data/models/faster_rcnn_resnet101/multiple_gpu
