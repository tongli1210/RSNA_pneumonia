r"""Convert raw RSNA dataset to TFRecord for object_detection.
Example usage:
    python create_rsna_tf_record.py \
        --data_dir= \ 
        --set= \
        --annotations_dir= \
        --label_csv= \
        --output_path= \
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import sys
import glob
import numpy as np

from lxml import etree
import PIL.Image
import tensorflow as tf
import pandas as pd
import pydicom 

#### Depend on tensorflow object_detection API
sys.path.insert(0, '/home/ltong/tensorflow/models/research')
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'ROOT directory to raw RSNA dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_csv', 'stage_1_train_labels.csv',
                    'file name for label csv file')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


def load_image(image_path):
    ds = pydicom.read_file(image_path) 
    image = ds.pixel_array
    # if grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    return image

def get_dicom_fps(data_dir):
    """
    Get all dicom image in the data directory
    
    Args:
        data_dir: the path to the data folder
    Returns:
        dicom_fps: the dicom image names
    """
    dicom_fps = glob.glob(data_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def load_mask(image_anns, height, width):
    # Initiation 
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
 
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)     
    count = len(image_anns)
    ret = 0  
    if count == 0:
        pass
    else:
        for a in image_anns:
            if a['Target'] == 1:
                ret = 1
                x = float(a['x'])
                y = float(a['y'])
                w = float(a['width'])
                h = float(a['height'])
                xmins.append(x/width)
                xmaxs.append((x+w)/width)
                ymins.append(y/height)
                ymaxs.append((y+h)/height)
                classes_text.append(b'Opacity')
                classes.append(1)

    if ret == 0:
        xmins.append(0)
        xmaxs.append(1.0)
        ymins.append(0)
        ymaxs.append(1.0)
        classes_text.append(b'Background')
        classes.append(0)

    return xmins, xmaxs, ymins, ymaxs, classes_text, classes

def parse_dataset(data_dir, anns):
    image_fps = get_dicom_fps(data_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(data_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations
   
def read_annotation_file(filename):
    """Reads RSNA annotation file.
    
    Args:
      filename: the path to the annotation csv file.

    Returns:
      anno: A dataframe of the annotation file
    """
    df_anns = pd.read_csv(filename)
    #print(df_anns.head())
    return df_anns

def create_tf_example(image_fp, image_anns, tmp_dir='/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/dataset_tools/tmpdata'):
    """
    Create a tf.Example proto from image
    
    Args:
        image instance
    
    Returns:
        example: the created tf.Example   
    
    """
    image = load_image(image_fp)
    height = image.shape[0]
    width = image.shape[1]
    for a in image_anns:
        image_id = a['patientId']
        break
    print(image_id)
    filename = os.path.join(tmp_dir, image_id+'.png')
      
    im = PIL.Image.fromarray(image)
    im.save(filename)
    
    filename = filename.encode() #encode 
    encoded_image_data = image.tostring() # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'
    
    xmins, xmaxs, ymins, ymaxs, classes_text, classes =  load_mask(image_anns, height, width)
    print(xmins, xmaxs, ymins, ymaxs, classes_text, classes)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
      
def convert_rsna_to_tfrecords(data_dir, annotations_dir, set_name, label_csv, output_path, sharding=False):
    """
    Convert RSNA dataset to TFRecords
    """ 
    writer = tf.python_io.TFRecordWriter(output_path)
    anns_csv_path = os.path.join(annotations_dir, label_csv)

    # Read in dataset to examples variable 
    # image_annotations: a dictionary with image_fp as the key 
    anns = read_annotation_file(anns_csv_path) 
    image_fps, image_annotations = parse_dataset(data_dir=data_dir, anns=anns)    

    if sharding:
        import contextlib2
        from google3.third_party.tensorflow_models.object_detection.dataset_tools import tf_record_creation_util
        num_shards=10
        output_filebase='/home/ltong/projects/Kaggle/RSNA_pneumonia/src/myFasterRCNN/data/rsna_'+set_name+'.record'
        
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_filebase, num_shards)
            for index, image_fp in image_fps:
                image_anns = image_annotations[image_fp]
                tf_example = create_tf_example(image_fp, image_anns)
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
   
    else:
        for image_fp in image_fps:
            image_anns = image_annotations[image_fp]
            tf_example = create_tf_example(image_fp, image_anns)        
            writer.write(tf_example.SerializeToString())

        writer.close()
    
def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))    
    convert_rsna_to_tfrecords(data_dir=FLAGS.data_dir,
                              annotations_dir=FLAGS.annotations_dir,
                              set_name=FLAGS.set,
                              label_csv=FLAGS.label_csv, 
                              output_path=FLAGS.output_path,
                              sharding=True)

if __name__ == "__main__":
    tf.app.run()
