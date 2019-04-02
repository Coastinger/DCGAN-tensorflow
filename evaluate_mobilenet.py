import numpy as np
import argparse
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from score_utils import mean_score, std_score

'''
Code taken from: https://github.com/titu1994/neural-image-assessment

Edited output and added total variation.

usage example:
python3 evaluate_mobilenet.py -dir=../dataset/out_sorted/Ukiyo-e/ -resize=True -resize_size=64 -rank=True
'''

parser = argparse.ArgumentParser(description='Evaluate NIMA(MobileNet)')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('-img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

parser.add_argument('-resize', type=str, default='false',
                    help='Resize images to 224x224 before scoring')
parser.add_argument('-resize_size', type=int, default='224',
                    help='Resize images to default 224x224 before scoring')

parser.add_argument('-rank', type=str, default='true',
                    help='Whether to rank the images after they have been scored')

args = parser.parse_args()
size = args.resize_size
print('resize_size: ', size)
resize_image = args.resize.lower() in ("true", "yes", "t", "1")
target_size = (size, size) if resize_image else None
rank_images = args.rank.lower() in ("true", "yes", "t", "1")

# give priority to directory
if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    imgs = Path(args.dir).files('*.png')
    imgs += Path(args.dir).files('*.jpg')
    imgs += Path(args.dir).files('*.jpeg')

elif args.img[0] is not None:
    print("Loading images from path(s) : ", args.img)
    imgs = args.img

else:
    raise RuntimeError('Either -dir or -img arguments must be passed as argument')

#with tf.device('/CPU:0'):
with tf.Session() as sess:
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')

    score_list = []
    std_list = []
    variation_list = []

    image = tf.placeholder(tf.float32, [None,size,size,3])
    total_var = tf.image.total_variation(image)

    mean_mean = 0
    std_mean = 0
    var_mean = 0

    for img_path in imgs:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

        mean_mean += mean
        std_mean += std

        x = (x + 1) / 2 # scale to [0,1]
        variation = sess.run([total_var], feed_dict={image: x})
        var_mean += variation[0]

        file_name = Path(img_path).name.lower()
        score_list.append((file_name, mean))
        std_list.append((file_name, std))
        variation_list.append((file_name, variation))

        print("Evaluating : ", img_path)
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        print("TV Score: ", variation)
        print()

    mean_mean /= len(score_list)
    std_mean /= len(std_list)
    var_mean /= len(variation_list)
    print('mean_mean: ', mean_mean)
    print('std_mean: ', std_mean)
    print('TV mean: ', var_mean)

    stddev_mean = 0
    for i, score in enumerate(score_list):
        stddev_mean += np.power(score[1] - mean_mean, 2)
    stddev_mean /= len(score_list)
    stddev_mean = np.sqrt(stddev_mean)
    print('stddev mean: ', stddev_mean)

    stddev_std = 0
    for i, score in enumerate(std_list):
        stddev_std += np.power(score[1] - std_mean, 2)
    stddev_std /= len(std_list)
    stddev_std = np.sqrt(stddev_std)
    print('stddev std: ', stddev_std)

    stddev_TV = 0.0
    for i, score in enumerate(variation_list):
        stddev_TV += np.power(score[1][0] - var_mean, 2)
    stddev_TV /= len(variation_list)
    stddev_TV = np.sqrt(stddev_TV)
    print('stddev TV: ', stddev_TV)

    if rank_images:
        print("*" * 40, "Ranking Images", "*" * 40)

        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(score_list[:10]):
            print('top mean: ', "%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))
        for i, (name, score) in enumerate(score_list[-10:]):
            print('low mean: ', "%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))

        std_list = sorted(std_list, key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(std_list[:10]):
            print('top std: ', "%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))
        for i, (name, score) in enumerate(std_list[-10:]):
            print('low std: ', "%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))

        variation_list = sorted(variation_list, key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(variation_list[:10]):
            print('top var: ', "%d)" % (i + 1), "%s : Score = %d" % (name, score[0]))
        for i, (name, score) in enumerate(variation_list[-10:]):
            print('low var: ', "%d)" % (i + 1), "%s : Score = %d" % (name, score[0]))
