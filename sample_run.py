
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import requests
import json

import ml_collections
import utils
from data.dataset import Dataset
from models import model as model_lib
from models import ar_model
from tasks import task as task_lib
from tasks import object_detection

# Define a Dataset class to use for finetuning.
class VocDataset(Dataset):

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Note: be consisous about 0 in label, which should probably reserved for
       special use (such as padding).

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels
    """


    feature_description = {
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
      }

    def _parse_function(example_proto):
      # Parse the input `tf.train.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, feature_description)

    parsed = _parse_function(example)
    dense_img = tf.sparse.to_dense(parsed['image/encoded'])
    # print(tf.image.decode_image(dense_img, dtype=tf.float32))
    decoded_img = tf.io.decode_jpeg(dense_img[0], channels = 3)


    features = {
        'image': tf.image.convert_image_dtype(decoded_img, tf.float32),
        'image/id': 0, # dummy int.
    }

    # The following labels are needed by the object detection task.
    label = tf.sparse.to_dense(parsed['image/object/class/label']) + 1  # 0 is reserved for padding.
    xmax = tf.sparse.to_dense(parsed['image/object/bbox/xmax'])
    xmin = tf.sparse.to_dense(parsed['image/object/bbox/xmin'])
    ymax = tf.sparse.to_dense(parsed['image/object/bbox/ymax'])
    ymin = tf.sparse.to_dense(parsed['image/object/bbox/ymin'])
    bbox = tf.stack([xmin, ymin, xmax, ymax], axis=1)
    # print(bbox)

    # Use tf.numpy_function to get features not easily computed in tf.
    def get_area(bboxes):
      return np.asarray([
          (b[2] - b[0]) * (b[3] - b[1]) for b in bboxes], dtype=np.float32)

    areas = tf.numpy_function(get_area, (bbox,), (tf.float32,))
    areas = tf.reshape(areas, [tf.shape(label)[0]])

    labels = {
        'label': label,
        # 'xmax': xmax,
        # 'xmin': xmin,
        # 'ymax': ymax,
        # 'ymin': ymin,
        'bbox': bbox,
        'area': areas,
        'is_crowd': tf.zeros_like(label, tf.bool),
    }


    # features = {
    #     'image': tf.image.convert_image_dtype(example['image'], tf.float32),
    #     'image/id': 0, # dummy int.
    # }

    # # The following labels are needed by the object detection task.
    # label = example['objects']['label'] + 1  # 0 is reserved for padding.
    # bbox = example['objects']['bbox']

    # # Use tf.numpy_function to get features not easily computed in tf.
    # def get_area(bboxes):
    #   return np.asarray([
    #       (b[2] - b[0]) * (b[3] - b[1]) for b in bboxes], dtype=np.int32)

    # areas = tf.numpy_function(get_area, (bbox,), (tf.int32,))
    # areas = tf.reshape(areas, [tf.shape(label)[0]])

    # labels = {
    #     'label': label,
    #     'bbox': bbox,
    #     'area': areas,
    #     'is_crowd': tf.zeros_like(label, tf.bool),
    # }
    
    return features, labels






# Load config for the pretrained model.
pretrained_model_dir = 'gs://pix2seq/obj365_pretrain/resnet_640x640_b256_s400k/'
with tf.io.gfile.GFile(os.path.join(pretrained_model_dir, 'config.json'), 'r') as f:
  config = ml_collections.ConfigDict(json.loads(f.read()))


# loaded_dataset = tf.data.TFRecordDataset("/content/drive/MyDrive/Matority/data/color_fashion_tfrec_train")


# Update config for finetuning (some configs were missing at initial pretraining time).
config.dataset.tfds_name = 'voc'
# config.dataset.data_dir = "/content/drive/MyDrive/Matority/data/color_fashion_tfrec_train"
config.dataset.batch_duplicates = 1
config.dataset.coco_annotations_dir = None
config.task.name == 'object_detection'
config.task.vocab_id = 10  # object_detection task vocab id.
config.task.weight = 1.
config.task.max_instances_per_image_test = 10
config.tasks = [config.task]
config.train.batch_size = 8
config.model.name = 'encoder_ar_decoder'  # name of model and trainer in registries.
config.model.pretrained_ckpt = pretrained_model_dir
config.optimization.learning_rate = 1e-4
config.optimization.warmup_steps = 10

# Use a smaller image_size to speed up finetuning here.
# You can use any image_size of choice.
config.model.image_size = 600
config.task.image_size = 600



# Perform training for 1000 steps. This takes about ~20 minutes on a regular Colab GPU.
train_steps = 100
use_tpu = False  # Set this accordingly.
steps_per_loop = 10
tf.config.run_functions_eagerly(False)

strategy = utils.build_strategy(use_tpu=use_tpu, master='')

# The following snippets are mostly copied and simplified from run.py.
with strategy.scope():
  # Get dataset.

  dataset = VocDataset(config)
  
  
  tmp_dataset = tf.data.TFRecordDataset("/content/drive/MyDrive/Matority/data/color_fashion_tfrec_train")
  num_train_examples = 0
  for i in tmp_dataset:
    num_train_examples += 1
  
  # Get task.
  task = task_lib.TaskRegistry.lookup(config.task.name)(config)
  tasks = [task]

  # Create tf.data.Dataset.
  ds = dataset.pipeline(
      process_single_example=task.preprocess_single,
      global_batch_size=config.train.batch_size,
      training=True)
  datasets = [ds]
  
  print("Data Pipeline Created!")
  
  # Setup training elements.
  trainer = model_lib.TrainerRegistry.lookup(config.model.name)(
      config, model_dir='model_dir',
      num_train_examples=num_train_examples, train_steps=train_steps)
  data_iterators = [iter(dataset) for dataset in datasets]

  print("Data Iterators Created!")

  @tf.function
  def train_multiple_steps(data_iterators, tasks):
    train_step = lambda xs, ts=tasks: trainer.train_step(xs, ts, strategy)
    for _ in tf.range(steps_per_loop):  # using tf.range prevents unroll.
      with tf.name_scope(''):  # prevent `while_` prefix for variable names.
        strategy.run(train_step, ([next(it) for it in data_iterators],))

  global_step = trainer.optimizer.iterations
  cur_step = global_step.numpy()
  while cur_step < train_steps:
    train_multiple_steps(data_iterators, tasks)
    cur_step = global_step.numpy()
    print(f"Done training {cur_step} steps.")