# This scripts sets up the development environment in a docker contanier

# load user specified configs
# ============================================================================
# EDIT THE VARIABLES BELOW:
# ============================================================================
export HOST_DATA_DIR=/home/jlin1/data/rsna
export HOST_PROJECT_DIR=/home/jlin1/videodev/rsna/detection_tf
# ============================================================================

# ============================================================================
# DO NOT TOUCH
export CLIENT_DATA_DIR=/data
export CLIENT_PROJECT_DIR=/detection
# ==========================================

export CONTAINER_NAME='rsna_detection'
export IMAGE_NAME=tensorflow_gpu:object_detection

# Start or unpause docker container
if docker ps | grep -q $CONTAINER_NAME; then
    echo Docker container already running
else
	if docker ps -a | grep -q $CONTAINER_NAME; then # unpause existing container
		echo Starting existing $CONTAINER_NAME container...
		nvidia-docker start $CONTAINER_NAME > /dev/null
	else
		echo Starting new $CONTAINER_NAME container...
    	nvidia-docker run --name $CONTAINER_NAME -td -p 8888:8888 -p 6006:6006 -v $HOST_PROJECT_DIR:$CLIENT_PROJECT_DIR -v $HOST_DATA_DIR:$CLIENT_DATA_DIR $IMAGE_NAME > /dev/null # start new container

    fi
fi

# Open bash prompt inside container
echo 'Opening bash prompt inside container'
nvidia-docker exec -i -t $CONTAINER_NAME /bin/bash
