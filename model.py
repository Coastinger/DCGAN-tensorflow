from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

from scipy.spatial.distance import cdist

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from score_utils import mean_score, std_score

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=32,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data',
         _lambda=1, use_can=False, use_slim_can=False, use_tiny_can=False, use_resize_conv=True,
         use_label_smoothing=True, G_optim_times=2):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.use_resize_conv = use_resize_conv
    self.use_label_smoothing = use_label_smoothing
    self.G_optim_times = G_optim_times

    self.batch_size = batch_size
    image_manifold_size(batch_size)
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self._lambda = _lambda

    self.use_can = use_can
    self.use_slim_can = use_slim_can
    self.use_tiny_can = use_tiny_can
    if self.use_slim_can or self.use_tiny_can:
        self.df_dim = 64

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    if not self.y_dim or self.use_slim_can or self.use_can or self.use_tiny_can:
      self.d_bn3 = batch_norm(name='d_bn3')
      if self.use_can:
          self.d_bn4 = batch_norm(name='d_bn4')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    if not self.y_dim or self.use_slim_can or self.use_can or self.use_tiny_can:
      self.g_bn3 = batch_norm(name='g_bn3')
      if self.use_can:
          self.g_bn4 = batch_norm(name='g_bn4')
          self.g_bn5 = batch_norm(name='g_bn5')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
      print('[SETUP] MNIST Input min/max: ',np.min(self.data_X[0]),np.max(self.data_X[0]))
    else:
      data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
      self.data = glob(data_path)
      self.label_dict = {}
      path_list = glob('../dataset/wikiart/**/', recursive=True)[1:]
      for i, elem in enumerate(path_list):
        self.label_dict[os.path.basename(os.path.normpath(elem))] = i
      print('[SETUP] y label dict: ', self.label_dict)
      if len(self.label_dict) != self.y_dim:
        raise Exception('[!] Missmatch of given y_dim ', str(self.y_dim), ' and found classes ', len(self.label_dict))
      if len(self.data) == 0:
        raise Exception("[!] No data found in '" + data_path + "'")
      np.random.shuffle(self.data)
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3:
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1
      if len(self.data) < self.batch_size:
        raise Exception("[!] Entire dataset size is less than the configured batch_size")

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs #TODO: with decay + tf.random.normal(shape=self.inputs.shape, mean=0.0, stddev=0.1, dtype=tf.float32)

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y, resize_conv=self.use_resize_conv)
    self.D, self.D_logits, self.D_c, self.D_c_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y, resize_conv=self.use_resize_conv)
    self.D_, self.D_logits_, self.D_c_, self.D_c_logits_ = self.discriminator(self.G, self.y, reuse=True)

    # MSSIM
    # NOTE: original mssim downsamples imgs 4 times, leads to problem if img size is below 176x176
    # if img size is lower, then cut power_factors, e.g. for 64x64 2 power_factors seems fine
    self.img1 = tf.placeholder(tf.float32, [None] + image_dims, name='mssim_img1')
    self.img2 = tf.placeholder(tf.float32, [None] + image_dims, name='mssim_img2')
    self.img1_yuv = tf.image.rgb_to_yuv(self.img1)
    self.img2_yuv = tf.image.rgb_to_yuv(self.img2)
    self.mssim = tf.image.ssim_multiscale(self.img1_yuv, self.img2_yuv, max_val=1.0, power_factors=[0.0448])#, 0.25856])
    self.mssim_ = scalar_summary('mssim_batch_mean', tf.reduce_mean(self.mssim))
    self.ssim = tf.image.ssim(self.img1_yuv, self.img2_yuv, max_val=1.0)
    self.ssim_sum = scalar_summary('ssim_batch_mean', tf.reduce_mean(self.ssim))

    # total_variation - for samples
    self.total_var = tf.image.total_variation(self.img1)
    self.total_var_sum = scalar_summary('total variation', tf.reduce_mean(self.total_var))

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.d_c_sum = histogram_summary("d_c", self.D_c)
    self.d_c__sum = histogram_summary("d_c_", self.D_c_)
    self.G_sum = image_summary("G", self.G)

    if self.y_dim:
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.D_c,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy_sum = scalar_summary("accuracy", self.accuracy)

    if self.use_label_smoothing:
        true_label = tf.random_uniform(tf.shape(self.D),.9, 1.0)
    else:
        true_label = 1.0

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D) * true_label))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    if self.use_can or self.use_slim_can or self.use_tiny_can:
      self.d_loss_class_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_c_logits,
          labels=self.y))
      self.g_loss_class_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_c_logits_,
          labels=(1.0/self.y_dim)*tf.ones_like(self.D_c_)))
      self.d_loss_class_real_sum = scalar_summary("d_loss_class_real", self.d_loss_class_real)
      self.g_loss_class_fake_sum = scalar_summary("g_loss_class_fake", self.g_loss_class_fake)

      self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_class_real
      self.g_loss = self.g_loss_fake + self._lambda * self.g_loss_class_fake
    else:
      self.d_loss = self.d_loss_real + self.d_loss_fake
      self.g_loss = self.g_loss_fake

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_fake = scalar_summary("g_loss_fake", self.g_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    if self.use_can or self.use_slim_can or self.use_tiny_can:
      self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, \
                        self.d_loss_class_real_sum, self.g_loss_class_fake_sum, self.accuracy_sum])
    else:
      self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs/log_" + self.model_dir + '_' + timestamp() + '/', self.sess.graph)

    if not os.path.exists('./samples/sample_z.npy'):
        print('[SETUP] No sample z. Creating...')
        sample_z = np.random.normal(0, 1, size=(self.sample_num , self.z_dim))
        np.save('./samples/sample_z.npy', sample_z)
    else:
        print('[SETUP] Found sample z. Loading...')
        sample_z = np.load('./samples/sample_z.npy')

    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      if not os.path.exists('./samples/sample_files.npy'):
          print('[SETUP] No sample files. Creating...')
          sample_files = self.data[0:self.sample_num]
          np.save('./samples/sample_files.npy', sample_files)
      else:
          print('[SETUP] Found sample files. Loading...')
          sample_files = np.load('./samples/sample_files.npy')
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      print('[SETUP] sample Input min/max: ',np.min(sample[0]),np.max(sample[0]))
      if self.y_dim:
        sample_labels = self.get_labels(sample_files)
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
      save_images(sample_inputs, image_manifold_size(sample_inputs.shape[0]), 'sample_inputs_preview.png')

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:
        #self.data = glob(os.path.join(config.data_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(self.data)
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size
        print('batch idx: ', batch_idxs)
      for idx in xrange(0, int(batch_idxs)):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.y_dim:
            batch_labels = self.get_labels(batch_files)
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)
          if epoch == 0 and idx == 0:
            print('Batch shape: ', batch_images.shape)
            save_images(batch_images, image_manifold_size(sample_inputs.shape[0]), 'batch_images_preview.png')

        batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]).astype(np.float32)
        #batch_z /= np.linalg.norm(batch_z, axis=0)

        if self.y_dim:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          for i in range(self.G_optim_times):
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={
                  self.z: batch_z,
                  self.y:batch_labels,
                })
              self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z,
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
          errD_class_real = self.d_loss_class_real.eval({
              self.inputs: batch_images,
              self.y: batch_labels
          })
          acc = self.accuracy.eval({
              self.inputs: batch_images,
              self.y: batch_labels
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        if self.use_can or self.use_slim_can or self.use_tiny_can:
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, acc: %.8f" \
              % (epoch, config.epoch, idx, batch_idxs,
                time.time() - start_time, errD_fake+errD_real+errD_class_real, errG, acc))
        else:
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, config.epoch, idx, batch_idxs,
                time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, batch_idxs) == 1: #batch_idxs
          if config.dataset == 'mnist' or 'wikiart':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

            # NIMA neural image quality assessment
            # https://github.com/titu1994/neural-image-assessment
            base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation='softmax')(x)
            model = Model(base_model.input, x)
            model.load_weights('weights/mobilenet_weights.h5')

            samples_scaled = samples.copy() #inverse_transform(sample_inputs)
            samples_exp = samples_scaled#preprocess_input(samples_scaled) # scale input to [-1,1] sample wise
            # nima batch, result differs by ~.02
            #scores = model.predict(samples_exp, batch_size=64, verbose=0)[0]
            #print(scores)
            #print('NIMA: ', mean_score(scores))

            mean_list = []
            std_list = []
            score_mul_list = []
            score_div_list = []
            mean_mean = 0
            std_mean = 0
            score_mul = 0
            score_div = 0
            for sample in samples_exp:
                x = sample #preprocess_input(sample)
                x = np.expand_dims(x, 0)
                scores = model.predict(x, batch_size=1, verbose=0)[0]
                #print(scores)
                mean = mean_score(scores)
                mean_mean += mean
                std = std_score(scores)
                std_mean += std
                #print('NIMA: ', mean, ' +- ', std)
                mean_list.append((mean, sample))
                std_list.append((std, sample))
                score = mean * std
                score_mul += score
                score_mul_list.append((score, sample))
                score = mean / std
                score_div += score
                score_div_list.append((score, sample))
            NIMA_mean = mean_mean / self.sample_num
            NIMA_std = std_mean / self.sample_num
            NIMA_score_mul = score_mul / self.sample_num
            NIMA_score_div = score_div / self.sample_num

            print('[Sample] NIMA mean: ', NIMA_mean)
            summary = tf.Summary(value=[tf.Summary.Value(tag='NIMA_mean',simple_value=NIMA_mean)])
            self.writer.add_summary(summary, counter)
            print('[Sample] NIMA std: ', NIMA_std)
            summary = tf.Summary(value=[tf.Summary.Value(tag='NIMA_std',simple_value=NIMA_std)])
            self.writer.add_summary(summary, counter)
            print('[Sample] NIMA mul(mean,std): ', NIMA_score_mul)
            summary = tf.Summary(value=[tf.Summary.Value(tag='NIMA_score_mul',simple_value=NIMA_score_mul)])
            self.writer.add_summary(summary, counter)
            print('[Sample] NIMA div(mean,std): ', NIMA_score_div)
            summary = tf.Summary(value=[tf.Summary.Value(tag='NIMA_score_div',simple_value=NIMA_score_div)])
            self.writer.add_summary(summary, counter)

            # sorted by NIMA: mean, std, mean * std, mean / std
            mean_list_sorted = sorted(mean_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in mean_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/nima_mean_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            std_list_sorted = sorted(std_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in std_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/nima_std_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            score_mul_list_sorted = sorted(score_mul_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in score_mul_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/nima_score_mul_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            score_div_list_sorted = sorted(score_div_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in score_div_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/nima_score_div_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))


            # total variation
            samples_scaled = inverse_transform(samples)
            total_var, summary_str = self.sess.run([self.total_var, self.total_var_sum], feed_dict={self.img1: samples_scaled})
            print('[Sample] total-variation mean: ', np.sum(total_var)/total_var.shape[0])
            self.writer.add_summary(summary_str, counter)
            total_var_list = []
            [total_var_list.append((total_var[i], samples[i])) for i, elem in enumerate(total_var)]
            total_var_list_sorted = sorted(total_var_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in total_var_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/total_variation_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

            # TV / std
            tv_std_list = []
            for i, score in enumerate(total_var_list):
                value = score[0] / std_list[i][0]
                tv_std_list.append((value, score[1]))
            tv_std_list_sorted = sorted(tv_std_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in tv_std_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/TV_std_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

            # TV / mean
            tv_mean_list = []
            for i, score in enumerate(total_var_list):
                value = score[0] / mean_list[i][0]
                tv_mean_list.append((value, score[1]))
            tv_mean_list_sorted = sorted(tv_mean_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in tv_mean_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/TV_mean_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

            # TV / div
            tv_div_list = []
            for i, score in enumerate(total_var_list):
                value = score[0] / score_div_list[i][0]
                tv_div_list.append((value, score[1]))
            tv_div_list_sorted = sorted(tv_div_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in tv_div_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/TV_div_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

            # TV / mul
            tv_mul_list = []
            for i, score in enumerate(total_var_list):
                value = score[0] / score_mul_list[i][0]
                tv_mul_list.append((value, score[1]))
            tv_mul_list_sorted = sorted(tv_mul_list, key=lambda x: x[0], reverse=True)
            sorted_samples = []
            for score in tv_mul_list_sorted:
                sorted_samples.append(score[1])
            sorted_samples = np.array(sorted_samples)
            save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
                  './{}/TV_mul_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

            # nearest neighbors
            print('Start n-nearest: ', time.time() - start_time)
            examples = 5
            neighbors = 5
            image_frame_dim = int(math.ceil(examples**.5))
            mse_ = 0
            mssim_ = 0
            combo_ = 0
            mse_imgs_list = []
            mssim_imgs_list = []
            combo_imgs_list = []
            for i in range(examples):
                sample = np.expand_dims(np.array(samples_scaled[i]),0)
                mse_sum, mssim_sum, combo_sum, nearest_imgs_mse, nearest_imgs_mssim, nearest_imgs_combo = \
                    self.nearest_neighbors(sample, samples, i, neighbors)
                #print('[Sample] nearest mssim: ', mse_sum)
                mse_ += mse_sum
                mssim_ += mssim_sum
                combo_ += combo_sum
                mse_imgs_list.append(nearest_imgs_mse)
                mssim_imgs_list.append(nearest_imgs_mssim)
                combo_imgs_list.append(nearest_imgs_combo)
            nearest_imgs_mse = np.reshape(np.array(mse_imgs_list), [examples*(neighbors+1),64,64,3])
            nearest_imgs_mssim = np.reshape(np.array(mssim_imgs_list), [examples*(neighbors+1),64,64,3])
            nearest_imgs_combo = np.reshape(np.array(combo_imgs_list), [examples*(neighbors+1),64,64,3])
            #print(nearest_imgs_mse.shape)
            save_images(nearest_imgs_mse, [examples, (neighbors+1)], \
                './{}/train_nearest_mssim_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            save_images(nearest_imgs_mssim, [examples, (neighbors+1)], \
                './{}/train_nearest_mse_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            save_images(nearest_imgs_combo, [examples, (neighbors+1)], \
                './{}/train_nearest_combo_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            mse_ /= examples
            mssim_ /= examples
            combo_ /= examples
            print('[Sample] 5-mean nearest mssim: ', mssim_)
            summary = tf.Summary(value=[tf.Summary.Value(tag='nearest_mssim',simple_value=mssim_)])
            self.writer.add_summary(summary, counter)
            print('[Sample] 5-mean nearest mse: ', mse_)
            summary = tf.Summary(value=[tf.Summary.Value(tag='nearest_mse',simple_value=mse_)])
            self.writer.add_summary(summary, counter)
            print('[Sample] 5-mean nearest combo: ', combo_)
            summary = tf.Summary(value=[tf.Summary.Value(tag='nearest_combo',simple_value=combo_)])
            self.writer.add_summary(summary, counter)
            print('End n-nearest: ', time.time() - start_time)

          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            except:
              print("one pic error!...")

        if np.mod(counter, batch_idxs*5) == 2:
          print('[Checkpoint] Saved.')
          self.save(config.checkpoint_dir, counter)
    # save final
    print('[Checkpoint] Final saved.')
    self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        print('D img no y: ',image)
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4, 1, 1
      elif self.use_can:
        print('D img CAN: ',image)
        if image.shape[1] < 64:
            h0 = image
            h1 = lrelu(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, name='d_h1_conv', padding='VALID'))
        else:
            h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, name='d_h0_conv', padding='VALID'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, k_h=4, k_w=4, name='d_h3_conv', padding='VALID')))
        h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, k_h=4, k_w=4, name='d_h4_conv', padding='VALID')))
        # real / fake
        h5 = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h5_lin')
        # style classification
        h6 = lrelu(linear(tf.reshape(h4, [self.batch_size, -1]), 1024, 'd_h6_lin'))
        h7 = lrelu(linear(h6, 512, 'd_h7_lin'))
        h8 = linear(h7, self.y_dim, 'd_h8_lin')

        return tf.nn.sigmoid(h5), h5, tf.nn.softmax(h8), h8
      elif self.use_slim_can:
        print('D img slim-CAN: ',image)
        h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, name='d_h0_conv', padding='VALID'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, k_h=4, k_w=4, name='d_h3_conv', padding='VALID')))
        # real / fake
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
        # style classification
        h5 = lrelu(linear(tf.reshape(h3, [self.batch_size, -1]), 1024, 'd_h5_lin'))
        h6 = lrelu(linear(h5, 512, 'd_h6_lin'))
        h7 = linear(h6, self.y_dim, 'd_h7_lin')

        return tf.nn.sigmoid(h4), h4, tf.nn.softmax(h7), h7
      elif self.use_tiny_can:
        print('D img slim-CAN: ',image)
        h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, name='d_h0_conv', padding='VALID'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        # real / fake
        h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')
        # style classification
        h4 = lrelu(linear(tf.reshape(h2, [self.batch_size, -1]), 512, 'd_h4_lin'))
        h5 = lrelu(linear(h4, 256, 'd_h5_lin'))
        h6 = linear(h5, self.y_dim, 'd_h6_lin')

        return tf.nn.sigmoid(h3), h3, tf.nn.softmax(h6), h6
      else:
        print('D img y: ',image)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        image = conv_cond_concat(image, yb)
        h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, name='d_h0_conv', padding='VALID'))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*4, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*8, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        shape = np.product(h2.get_shape()[1:].as_list())
        h2 = tf.reshape(h2, [-1, shape])
        h2 = concat([h2,y],1)
        # real / fake
        r_out = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_ro_lin')
        # style classification
        h3 = lrelu(linear(h2, 1024, 'd_h3_lin'))
        h4 = lrelu(linear(h3, 512, 'd_h4_lin'))
        c_out = linear(h4, self.y_dim, 'd_co_lin')

        return tf.nn.sigmoid(r_out), r_out, tf.nn.softmax(c_out), c_out

  def generator(self, z, y=None, resize_conv=True):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim or self.use_slim_can or self.use_tiny_can:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        if resize_conv:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            self.h1, self.h1_w, self.h1_b = resizeconv(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            h2, self.h2_w, self.h2_b = resizeconv(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3, self.h3_w, self.h3_b = resizeconv(h2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
            h4, self.h4_w, self.h4_b = resizeconv(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
        else:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

      elif self.use_can:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
        s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)

        if resize_conv:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(self.z_, [-1, s_h64, s_w64, self.gf_dim * 16])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            self.h1, self.h1_w, self.h1_b = resizeconv(h0, [self.batch_size, s_h32, s_w32, self.gf_dim*16], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            h2, self.h2_w, self.h2_b = resizeconv(h1, [self.batch_size, s_h16, s_w16, self.gf_dim*8], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3, self.h3_w, self.h3_b = resizeconv(h2, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
            h4, self.h4_w, self.h4_b = resizeconv(h3, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))
            h5, self.h5_w, self.h5_b = resizeconv(h4, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5))
            h6, self.h6_w, self.h6_b = resizeconv(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)
        else:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(self.z_, [-1, s_h64, s_w64, self.gf_dim * 16])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h32, s_w32, self.gf_dim*16], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h16, s_w16, self.gf_dim*8], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))
            h5, self.h5_w, self.h5_b = deconv2d(h4, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5))
            h6, self.h6_w, self.h6_b = deconv2d(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)

        return tf.nn.tanh(h6)

      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def sampler(self, z, y=None, resize_conv=True):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim  or self.use_can:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
        s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)

        if resize_conv:
            self.z_ = linear(z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin')
            self.h0 = tf.reshape(self.z_, [-1, s_h64, s_w64, self.gf_dim * 16])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=False))
            self.h1 = resizeconv(h0, [self.batch_size, s_h32, s_w32, self.gf_dim*16], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=False))
            h2 = resizeconv(h1, [self.batch_size, s_h16, s_w16, self.gf_dim*8], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))
            h3 = resizeconv(h2, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))
            h4 = resizeconv(h3, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4, train=False))
            h5 = resizeconv(h4, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h5')
            h5 = tf.nn.relu(self.g_bn5(h5, train=False))
            h6 = resizeconv(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6')
        else:
            self.z_ = linear(z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin')
            self.h0 = tf.reshape(self.z_, [-1, s_h64, s_w64, self.gf_dim * 16])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=False))
            self.h1 = deconv2d(h0, [self.batch_size, s_h32, s_w32, self.gf_dim*16], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=False))
            h2 = deconv2d(h1, [self.batch_size, s_h16, s_w16, self.gf_dim*8], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))
            h3 = deconv2d(h2, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))
            h4 = deconv2d(h3, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4, train=False))
            h5 = deconv2d(h4, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h5')
            h5 = tf.nn.relu(self.g_bn5(h5, train=False))
            h6 = deconv2d(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6')

        return tf.nn.tanh(h6)

      elif self.use_slim_can or self.use_tiny_can:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        if resize_conv:
            h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))
            h1 = resizeconv(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))
            h2 = resizeconv(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))
            h3 = resizeconv(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))
            h4 = resizeconv(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
        else:
            h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))
            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))
            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))
            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))
            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)

      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)
        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)
        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)
        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    return X/255.*2-1,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  def get_labels(self, inputs):
    labels = []
    for sample in inputs:
      _, _, _, lab_str, _ = sample.split('/', 4)
      try:
          labels.append(np.eye(self.y_dim)[np.array(self.label_dict[lab_str])])
      except IndexError:
          print('[!] IndexError - probably unmatching y_dim and provided folders.')
    return labels

  def nearest_neighbors(self, sample, samples, example, neighbors):
      # n-nearest-neighbors (MSSIM and MSE)
      #print('Start n-nearest: ', time.time() - start_time)
      #sample = np.expand_dims(np.array(samples_scaled[0]),0)
      data_path = '../dataset_npy/wikiart/**/*.npy'
      data = glob(data_path)
      distances_mssim = []
      distances_mse = []
      distances_combo = []
      #distance_ssim = []
      zero_dist = 0
      max_mse = l2(np.ones([1,64,64,3]), np.zeros([1,64,64,3]))
      min_mse = 1 # l2(np.zeros([1,64,64,3]), np.ones([1,64,64,3])) would be same as max, np.0.99() = 1.108
      #print('max_se: ',max_mse)
      #print('min_mse: ', min_mse)
      for i, path in enumerate(data):
          img = np.load(path)
          img = np.expand_dims(img,0)
          img = inverse_transform(img)
          #ssim, _ = self.sess.run([self.ssim, self.ssim_sum], feed_dict={self.img1: sample, self.img2: img})
          #distance_ssim.append((ssim, path))
          mssim, summary_str = self.sess.run([self.mssim, self.mssim_], feed_dict={self.img1: sample, self.img2: img})
          distances_mssim.append((mssim, path, summary_str))
          mse = l2(sample, img)
          distances_mse.append((mse, path))
          dist = l2((mse-min_mse)/(max_mse-min_mse), mssim) # (mse / max_mse) + mssim # TODO factoren
          distances_combo.append((dist, path))

      distances_mssim = sorted(distances_mssim, reverse=True) # higher = better
      distances_mse = sorted(distances_mse)
      distances_combo = sorted(distances_combo, reverse=True) # higher = better
      #distance_ssim = sorted(distance_ssim)

      nearest_mssim = distances_mssim[:neighbors]
      nearest_mse = distances_mse[:neighbors]
      nearest_combo = distances_combo[:neighbors]

      print('[Sample] nearest mssim: ', nearest_mssim[0][0])
      #summary = tf.Summary(value=[tf.Summary.Value(tag='nearest_mssim',simple_value=nearest_mssim[0][0])])
      #self.writer.add_summary(summary, counter)
      print('[Sample] nearest mse: ', nearest_mse[0][0])
      #summary = tf.Summary(value=[tf.Summary.Value(tag='nearest_mse',simple_value=nearest_mse[0][0])])
      #self.writer.add_summary(summary, counter)
      print('[Sample] nearest combo: ', nearest_combo[0][0])
      #summary = tf.Summary(value=[tf.Summary.Value(tag='nearest_combo',simple_value=nearest_combo[0][0])])
      #self.writer.add_summary(summary, counter)

      nearest_imgs_mssim = []
      for elem in nearest_mssim:
          #print('mssim: ',elem[0])
          img = np.load(elem[1]) # double load, yes, but otherwise sort is complicated
          nearest_imgs_mssim.append(img)
      nearest_imgs_mse = []
      for elem in nearest_mse:
          #print('mse: ',elem[0])
          img = np.load(elem[1])
          nearest_imgs_mse.append(img)
      nearest_imgs_combo = []
      for elem in nearest_combo:
          #print('mssim_mse: ',elem[0])
          img = np.load(elem[1])
          nearest_imgs_combo.append(img)
      nearest_imgs_mssim.insert(0, np.array(samples[example]))
      nearest_imgs_mse.insert(0, np.array(samples[example]))
      nearest_imgs_combo.insert(0, np.array(samples[example]))
      nearest_imgs_mssim = np.array(nearest_imgs_mssim)
      nearest_imgs_mse = np.array(nearest_imgs_mse)
      nearest_imgs_combo = np.array(nearest_imgs_combo)

      return nearest_mse[0][0], nearest_mssim[0][0], nearest_combo[0][0], nearest_imgs_mse, nearest_imgs_mssim, nearest_imgs_combo

      #save_images(nearest_imgs_mssim, [1, len(nearest_imgs_mssim)], './{}/train_nearest_mssim_%s_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx) % name)
      #save_images(nearest_imgs_mse, [1, len(nearest_imgs_mse)], './{}/train_nearest_mse_{:02d}_%s_{:04d}.png'.format(sample_dir, epoch, idx)  % name)
      #save_images(nearest_imgs_combo, [1, len(nearest_imgs_combo)], './{}/train_nearest_combo_%s_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx) % name)
