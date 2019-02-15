import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "wikiart", "The name of dataset [celebA, mnist, lsun, wikiart]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("G_optim_times", 2, "How many times G is optimised per batch. [2]")
flags.DEFINE_integer("y_dim", 10, "The size of provided labels. [10]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("use_can", False, "Use CAN implementation. [False]")
flags.DEFINE_boolean("use_slim_can", False, "Use slim CAN implementation. [False]")
flags.DEFINE_boolean("use_tiny_can", False, "Use tiny CAN implementation. [False]")
flags.DEFINE_float('_lambda', 1, 'Scalar to controll style ambiguity of G loss.')
flags.DEFINE_boolean('use_resize_conv', False, 'If True use resize_conv otherwise deconv2d. [False]')
flags.DEFINE_boolean('use_label_smoothing', False, 'If True use label smoothing [0.9 to 1.0] otherwise not. [False]')
FLAGS = flags.FLAGS

def main(_):
  #pp.pprint(flags.FLAGS.__flags)
  for key in tf.app.flags.FLAGS.flag_values_dict():
    print(key,'   :   ', FLAGS[key].value)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist' or FLAGS.dataset == 'wikiart':
      print('[SETUP] Creating wikiart DCGAN.')
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=FLAGS.y_dim,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir,
          use_can=FLAGS.use_can,
          use_slim_can=FLAGS.use_slim_can,
          use_tiny_can=FLAGS.use_tiny_can,
          _lambda=FLAGS._lambda,
          use_resize_conv=FLAGS.use_resize_conv,
          use_label_smoothing=FLAGS.use_label_smoothing,
          G_optim_times=FLAGS.G_optim_times)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir,
          use_can=FLAGS.use_can)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    # Below is codes for visualization
    # 0 = random batch
    # 1 = interpolation between -1 and 1
    # 2 = interpolation between 2 random vecs
    # 3 = chose 4 interpol corners
    OPTION = 0
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
