"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import datetime
import pickle
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim

from scipy.spatial.distance import cdist

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from score_utils import mean_score, std_score

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  try:
      return transform(image, input_height, input_width, resize_height, resize_width, crop)
  except ValueError:
      print('Corrupted Image. Path: ', image_path)

def center_and_norm(x):
    x = (x - np.mean(x)) / np.std(x)
    x = 2 * ((x - np.min(x))/(np.max(x) - np.min(x))) - 1
    #print('normalization output min/max: ',np.min(x[0]),np.max(x[0]))
    return x

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def l2(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))

def visualize(sess, dcgan, config, option):
  print('visualize...')
  image_frame_dim = int(math.ceil(config.batch_size**.5))

  if option == 0:
    print('option 0, one random test batch.')
    #z_sample = np.load('./samples/sample_z.npy')
    z_sample = np.random.normal(0, 1, size=(config.batch_size, dcgan.z_dim))
    #z_sample /= np.linalg.norm(z_sample, axis=0)
    #pickle.dump(z_sample, open('./samples/z_sample_%s.p' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()), 'wb'))
    #print('z_sample: ', z_sample)
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

    # total variation
    samples_scaled = inverse_transform(samples)
    total_var, summary_str = sess.run([dcgan.total_var, dcgan.total_var_sum], feed_dict={dcgan.img1: samples_scaled})
    total_var_mean = np.sum(total_var)/total_var.shape[0]
    print('[Sample] mean total variation: ', total_var_mean)
    total_var_list = []
    [total_var_list.append((total_var[i], samples[i])) for i, elem in enumerate(total_var)]
    total_var_list_sorted = sorted(total_var_list, key=lambda x: x[0], reverse=True)
    sorted_samples = []
    for score in total_var_list_sorted:
        #print('total var: ', score[0])
        sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
          './samples/test_%s_total_var.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

    # NIMA neural image quality assessment
    # https://github.com/titu1994/neural-image-assessment
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')

    samples_scaled = samples.copy() #inverse_transform(sample_inputs)
    samples_exp = preprocess_input(samples_scaled) #np.expand_dims(samples, 0)

    mean_list = []
    std_list = []
    score_mul_list = []
    score_div_list = []
    score_dist_list = []
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
        score_dist = l2(mean, std)
        score_dist_list.append((score_dist, sample))
    NIMA_mean = mean_mean / len(samples_exp)
    NIMA_std = std_mean / len(samples_exp)
    NIMA_score_mul = score_mul / len(samples_exp)
    NIMA_score_div = score_div / len(samples_exp)

    # combo var + std
    combo_var_mean = []
    for i, var in enumerate(total_var_list):
        x = var[0] + std_list[i][0]
        combo_var_mean.append((x, var[1]))
    # combo var + std dann mean sort
    combo_var_mean_sorted = sorted(combo_var_mean, key=lambda x: x[0], reverse=True)
    sorted_samples = []
    for score in combo_var_mean_sorted:
        sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
          './samples/test_%s_combo_var_std.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

    # combo dist(mean, std)
    combo_std_mean = []
    for i, mean in enumerate(mean_list):
        x = l2(mean[0], std_list[i][0])
        combo_std_mean.append((x, mean[1]))
    combo_var_mean_sorted = sorted(combo_std_mean, key=lambda x: x[0], reverse=True)
    sorted_samples = []
    for score in combo_var_mean_sorted:
        sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
        './samples/test_%s_dist_std_mean.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

    # filter with var
    total_var_list.clear()
    for i, elem in enumerate(total_var):
      if elem < total_var_mean and elem > total_var_mean/2:
          total_var_list.append((total_var[i], samples[i], mean_list[i][0], score_dist_list[i][0], score_div_list[i][0], score_mul_list[i][0]))
    # sort by var
    total_var_list_sorted = sorted(total_var_list, key=lambda x: x[0])
    print('# after var filtering: ',len(total_var_list_sorted))
    while not np.sqrt(len(total_var_list_sorted)).is_integer():
        total_var_list_sorted.pop()
    print('# after var filtering: ',len(total_var_list_sorted))
    sorted_samples = []
    for score in total_var_list_sorted:
      sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
        './samples/test_%s_total_var_below_mean.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    # sort by mean
    total_var_list_sorted.clear()
    total_var_list_sorted = sorted(total_var_list, key=lambda x: x[2], reverse=True)
    while not np.sqrt(len(total_var_list_sorted)).is_integer():
        total_var_list_sorted.pop()
    sorted_samples = []
    for score in total_var_list_sorted:
      sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
        './samples/test_%s_total_var_below_mean_mean.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    # sort by dist(mean,std)
    total_var_list_sorted.clear()
    total_var_list_sorted = sorted(total_var_list, key=lambda x: x[3], reverse=True)
    while not np.sqrt(len(total_var_list_sorted)).is_integer():
        total_var_list_sorted.pop()
    sorted_samples = []
    for score in total_var_list_sorted:
      sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
        './samples/test_%s_total_var_below_mean_dist.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    # sort by div
    total_var_list_sorted.clear()
    total_var_list_sorted = sorted(total_var_list, key=lambda x: x[4], reverse=True)
    while not np.sqrt(len(total_var_list_sorted)).is_integer():
        total_var_list_sorted.pop()
    sorted_samples = []
    for score in total_var_list_sorted:
      sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
        './samples/test_%s_total_var_below_mean_div.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    # sort by mul
    total_var_list_sorted.clear()
    total_var_list_sorted = sorted(total_var_list, key=lambda x: x[5], reverse=True)
    while not np.sqrt(len(total_var_list_sorted)).is_integer():
        total_var_list_sorted.pop()
    sorted_samples = []
    for score in total_var_list_sorted:
      sorted_samples.append(score[1])
    sorted_samples = np.array(sorted_samples)
    save_images(sorted_samples, image_manifold_size(sorted_samples.shape[0]),
        './samples/test_%s_total_var_below_mean_mul.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))



  elif option == 1:
    print('option 1, interpolation from constant -1 to 1.')
    values = np.arange(-1, 1, 2./config.batch_size)
    print('values shape: ', values.shape)
    z_sample = np.ones([config.batch_size, dcgan.z_dim])
    #print(z_sample)
    for idx, z in enumerate(z_sample):
        z_sample[idx] = z * values[idx]
        #s_inter = slerp(values[idx], -1, 1)
        #print('spherical test: ', values[idx], ' -> ', s_inter)
    #print(z_sample)
    #z_sample /= np.linalg.norm(z_sample, axis=0)
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_inter_oneTOone_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

  elif option == 2:
    print('option 2, interpolation between two random vecs.')
    z_sample = np.random.normal(0, 1, size=(2, dcgan.z_dim))
    print('z_samples: ', z_sample[0], z_sample[1])
    way = z_sample[0] - z_sample[1]
    mean_dist = np.sum(way) / dcgan.z_dim
    print('mean dist: ', mean_dist)
    way = way / (config.batch_size-1)
    inter = np.tile(z_sample[0], [config.batch_size, 1])
    for i, elem in enumerate(inter):
        inter[i] = elem - i * way
    inter /= np.linalg.norm(inter, axis=0)
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: inter})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_inter_2random_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

  elif option == 3:
    # bilinear interpolation
    # https://dsp.stackexchange.com/questions/13697/how-do-you-interpolate-between-points-in-an-image-2d-e-g-using-splines
    print('option 3, interpolation between 4 chosen corners.')
    # change according to filename
    #z_sample = pickle.load(open('./samples/wikiart_0127/z_sample_2019-01-28-12-52-55.p', 'rb'))
    z_sample = np.load('./samples/sample_z.npy') # pickle.load(open('./samples/wikiart_0128_64_resize/z_sample_2019-01-30-11-41-52.p', 'rb'))

    '''
    # get length with norm
    norm = np.linalg.norm(z_sample, axis=1)
    sorted_norm = norm.argsort()
    get_last_norms = sorted_norm[-4:]
    A = z_sample[get_last_norms[0]]
    B = z_sample[get_last_norms[1]]
    C = z_sample[get_last_norms[2]]
    D = z_sample[get_last_norms[3]]
    '''
    '''
    # create dist matrix
    dist_matrix = cdist(z_sample, np.zeros(z_sample.shape), 'euclidean')
    print('dist_matrix: ', dist_matrix)
    print(dist_matrix[0].shape)
    print(dist_matrix[0].argsort())
    three_far_away = dist_matrix[0].argsort()[-3:]
    A = z_sample[0]
    B = z_sample[three_far_away[0]]
    C = z_sample[three_far_away[1]]
    D = z_sample[three_far_away[2]]
    '''
    # pick corners
    A = z_sample[0]
    B = z_sample[1]
    C = z_sample[3]
    D = z_sample[4]

    # save corner samples
    chosen = np.reshape(np.concatenate((A,B,C,D)), [4,-1])
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: chosen})
    save_images(samples, [2, 2], './samples/test_4_chosen_corners_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    # create interpolation grid
    # NOTE: chosen corners are not displayed on grid
    grid = np.ones([config.batch_size, dcgan.z_dim])
    x1 = y1 = 0
    x2 = y2 = image_frame_dim
    for i, elem in enumerate(grid):
        x = i % image_frame_dim
        y = i // image_frame_dim
        grid[i] = A * (((x2-x)*(y2-y))/((x2-x1)*(y2-y1))) \
                + B * (((x-x1)*(y2-y))/((x2-x1)*(y2-y1))) \
                + C * (((x2-x)*(y-y1))/((x2-x1)*(y2-y1))) \
                + D * (((x-x1)*(y-y1))/((x2-x1)*(y2-y1)))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: grid})
    print(samples.shape)
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_inter_4_chosen_corners_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

  elif option == 4:
    print('option 4, nearest neighbors in dataset')
    z_sample = pickle.load(open('./samples/wikiart_0127/z_sample_2019-01-28-12-52-55.p', 'rb'))
    chosen = z_sample[3]
    chosen = np.reshape(chosen, [1,-1])
    sample = sess.run(dcgan.sampler, feed_dict={dcgan.z: chosen})
    sample = np.reshape(sample, [32,32,3])
    #print('sample shape: ', sample.shape)
    #print('sample: ', sample)
    #print(np.min(sample), np.max(sample))

    data_path = '../dataset/wikiart/**/*.jpg'
    data = glob(data_path)
    print('data len: ', len(data))

    last_dist = float('Inf')
    nearest_neighbors = []
    for path in data:
        img = get_image(path,32,32,32,32,False,False)
        #print(np.min(img), np.max(img))
        #break
        dist = np.linalg.norm((sample.flatten() - img.flatten()), ord=1)
        if dist < last_dist:
            nearest_neighbors.append(img)
            last_dist = dist
            if len(nearest_neighbors) > 5:
                nearest_neighbors.pop()
    nearest_neighbors.append(sample)
    nearest_neighbors = np.reshape(np.array(nearest_neighbors), [len(nearest_neighbors), 32,32,3])

    save_images(nearest_neighbors, [1, len(nearest_neighbors)], './samples/test_nearest_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

  elif option == 5:
    print('option 5, create dataset mean image')
    data_path = '../dataset/wikiart/**/*.jpg'
    data = glob(data_path)
    print('data len: ', len(data))
    mean_img = np.zeros([32,32,3])
    for path in data:
        img = get_image(path,32,32,32,32,False,False)
        mean_img += img
    mean_img /= len(data)
    print(mean_img.shape)
    save_images(np.expand_dims(mean_img, 0), [1,1], 'mean.png')

# github: dribnet/plat - spherical interpolation (with boundarie [-1,1])
def slerp(val, low, high):
  if val <= -1:
      return low
  elif val >= 1:
      return high
  elif np.allclose(low,high):
      return low
  omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
  so = np.sin(omega)
  return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

def timestamp(format='%Y_%m_%d_%H_%M_%S'):
    """Returns the current time as a string."""
    return datetime.datetime.now().strftime(format)
