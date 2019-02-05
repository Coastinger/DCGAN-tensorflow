import numpy as np
from glob import glob
import os
import argparse
import utils

def main():

    if not os.path.exists(args.save_dir):
        print('[Info] Create destination dir. ', args.save_dir)
        os.makedirs(args.save_dir)

    splitted_path = []

    data_path = os.path.join(args.data_dir)
    path_list = glob(data_path)
    for i, path in enumerate(path_list):
        img = utils.get_image(path,args.size, args.size, args.size, args.size, crop=False, grayscale=False) # crop / resize
        img_array = np.array(img).astype(np.float32)
        split_path = path.split('/')
        new_base_path = os.path.join(split_path[0], args.save_dir, split_path[2], split_path[3])
        if not os.path.exists(new_base_path):
            os.makedirs(new_base_path)
        new_path = os.path.join(new_base_path, os.path.basename(path).split('.')[0]) + '.npy'
        np.save(new_path, img_array)
        print(str(i),' of ', len(path_list))

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', required=False, dest='data_dir', default='../dataset/wikiart/**/*.jpg',
                help='dir with source images - e.g. ./data/*.jpg')
ap.add_argument('--save_dir', required=False, dest='save_dir', default='dataset_npy',
                help='destination dir (only name) to save .npy files')
ap.add_argument('--size', required=False, dest='size', default=64,
                help='output size')

args = ap.parse_args()

if __name__ == '__main__':
    main()
