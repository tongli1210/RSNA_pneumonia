FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER xxxx

RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y wget unzip \
	git \
	python-pip \
	python-tk \
	pkg-config \
	zip \
	g++ \
	zlib1g-dev  \
	python-pil \
	python-lxml \
	vim

RUN pip install --upgrade pip

RUN pip install	scikit-learn \
	pydicom \
	Pillow \
	h5py \
	keras_applications==1.0.5 \
	keras_preprocessing==1.0.3 \
	matplotlib \
	numpy \
	pandas \
	scipy \
	scikit-learn \
	Cython \
	ipykernel \
	contextlib2

RUN python -m ipykernel.kernelspec

# install tensorflow
RUN mkdir /tensorflow
RUN pip install tensorflow-gpu

# install tensorflow object detection API
RUN mkdir /tensorflow/models
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

RUN mkdir /cocoapi
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR cocoapi/PythonAPI
RUN make
RUN cp -r pycocotools /tensorflow/models/research/

RUN mkdir /protobuf
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip -d /protobuf
WORKDIR /tensorflow/models/research
RUN /protobuf/bin/protoc object_detection/protos/*.proto --python_out=.

RUN pip install opencv-python
RUN pip install opencv-contrib-python

RUN apt install -y libsm6 libxext6

RUN pip install jupyter
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888
EXPOSE 6006

ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

WORKDIR /

CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tensorflow/models/research/object_detection", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

