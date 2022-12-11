#CUDA_VISIBLE_DEVICES=0 python main_mask_L3DAS.py B_format train 12BB01 ['theta','MV'] '' '' 
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tf debugging messages
import time
from train_networks import train
from create_args import *

from dotenv import find_dotenv, dotenv_values

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.__version__)
print(tf.config.list_physical_devices())

LD_LIBRARY_PATH='/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64'
LD_INCLUDE_PATH='/usr/local/cuda/include:/usr/local/cuda/extras/CUPTI/include'


def main_mask():

    # train
    train(args, config)

    print("Done!",time.ctime())


if __name__ == '__main__':

    args, config = create_args()

    # create paths
    # audio paths
    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)

    if not os.path.exists(args.audio_challenge_mixtures_path):
        os.makedirs(args.audio_challenge_mixtures_path)

    if not os.path.exists(args.audio_challenge_clean_path):
        os.makedirs(args.audio_challenge_clean_path)

    if not os.path.exists(args.audio_challenge_separated_path):
        os.makedirs(args.audio_challenge_separated_path)

    if not os.path.exists(args.audio_challenge_separated_gt_path):
        os.makedirs(args.audio_challenge_separated_gt_path)


    # figures paths
    if not os.path.exists(args.fig_challenge_mixtures_path):
        os.makedirs(args.fig_challenge_mixtures_path)

    if not os.path.exists(args.fig_challenge_clean_path):
        os.makedirs(args.fig_challenge_clean_path)

    if not os.path.exists(args.fig_challenge_separated_path):
        os.makedirs(args.fig_challenge_separated_path)

    if not os.path.exists(args.fig_challenge_masks_path):
        os.makedirs(args.fig_challenge_masks_path )

    if not os.path.exists(args.fig_challenge_separated_gt_path):
        os.makedirs(args.fig_challenge_separated_gt_path )

    main_mask()

