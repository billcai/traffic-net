import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
import argparse
import skimage, skimage.io

import datetime
import pytz
import pandas as pd

VEHICLES = [3,4,7,8]
THRESHOLD = 0.3
AREA_THRESHOLD = 0.2

def load_tf_model(model_loc):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(model_loc, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def load_image(image_path):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    return image_np

def load_and_eval(image_path,graph):
    image_np = load_image(image_path)
    output_dict = run_inference_for_single_image(image_np,graph)
    return output_dict

def filter_output_dict(output_dict,threshold,area_threshold,vehicles):
    output_dict['area'] = [
        ((i[2]-i[0])*(i[3]-i[1]))
        for i in output_dict['detection_boxes']
    ]
    vehicle_index = [x for x in range(len(output_dict['detection_classes']))
                if (output_dict['detection_classes'][x] in VEHICLES and
                output_dict['detection_scores'][x]>threshold and
                output_dict['area'][x] < area_threshold
                ) ]
    new_dict = {}
    for i in ['detection_classes','detection_scores','detection_boxes']:
        new_dict[i] = output_dict[i][vehicle_index]
    new_dict['num_detections'] = len(vehicle_index)
    return new_dict

def save_output_image(output_dict,original_image,threshold,output_location):
    image = vis_util.visualize_boxes_and_labels_on_image_array(original_image,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      use_normalized_coordinates=True,skip_scores=True,min_score_thresh=threshold,
      line_thickness=2)
    skimage.io.imsave(output_location,np.divide(image*1.0,255))
    return image

def update_logs(sg_time,filename,output_dict,record_folder,all_records,all_records_name):
    log_filename = os.path.join(record_folder,'.'.join(filename.split('.')[:-1])+'.csv')
    num_vehicles = output_dict['num_detections']
    print(str(num_vehicles))
    new_row = {'datetime':sg_time,'num_vehicles':num_vehicles}
    if os.path.isfile(log_filename):
        cur_log = pd.read_csv(log_filename)
        cur_log = cur_log.append(new_row,ignore_index=True)
    else:
        cur_log = pd.DataFrame(new_row,index=[0])
    cur_log.to_csv(log_filename,index=False)
    search_name = '.'.join(filename.split('.')[:-1])
    if np.sum(all_records['CameraID']==int(search_name)) == 1:
        print("Found and updating logs for "+search_name)
        all_records.loc[all_records['CameraID']==int(search_name),'num_vehicles'] = num_vehicles
    else:
        print("ERROR found "+str(np.sum(all_records['CameraID']==int(search_name)))+" records for "+search_name)
    all_records.to_csv(all_records_name,index=False)
    return all_records

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-model','--model',help="Model location")
  parser.add_argument('-input_folder','--input_folder',help ="input folder")
  parser.add_argument('-output_folder','--output_folder',help ="output folder")
  parser.add_argument('-record_folder','--record_folder',help ="record folder")
  parser.add_argument('-all_records','--all_records',help ="record folder")
  args = parser.parse_args()

  category_index = label_map_util.create_category_index_from_labelmap('./mscoco_label_map.pbtxt', use_display_name=True)
  model = load_tf_model(args.model)
  eval_files = [os.path.join(args.input_folder,x) for x in os.listdir(args.input_folder) if x[-4:]=='.jpg']
  output_list = []
  all_df = pd.read_csv(args.all_records)
  cur_time = datetime.datetime.utcnow()
  cur_time = cur_time.replace(tzinfo=pytz.utc)
  sg_time = (cur_time.astimezone(pytz.timezone('Singapore'))).strftime('%Y-%m-%d %H:%M:%S')
  for i in eval_files:
      print(i)
      output_dict = load_and_eval(i,model)
      output_dict = filter_output_dict(output_dict,THRESHOLD,AREA_THRESHOLD,VEHICLES)
      basename_file = os.path.basename(i)
      original_image = load_image(i)
      if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)
      save_output_image(output_dict,original_image,THRESHOLD,os.path.join(args.output_folder,basename_file))
      all_df = update_logs(sg_time,basename_file,output_dict,args.record_folder,all_df,args.all_records)
