import sys
import numpy as np
import os
import time

from dotenv import find_dotenv, dotenv_values
import yaml
import argparse


def create_args():

    # parse arguments
    parser = argparse.ArgumentParser(description="Training.")
    args = parser.parse_args()

    # load path variables
    loc_env = find_dotenv("paths.env")
    config_env = dotenv_values(loc_env)

    # set main paths
    args.WORK_PATH = config_env["WORK_PATH"]
    args.CONFIG_PATH = os.path.join(args.WORK_PATH, 'config.yaml')

    # load config file
    with open(args.CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    print(config)

    # config to args
    args.challenge_name = config['arguments']['challenge_name']
    args.case_name = config['arguments']['case_name']
    args.task = config['arguments']['task']
    args.Train_Room = config['arguments']['Train_Room'] 
    args.theta_feature = config['arguments']['theta_feature'] 
    args.LPS_feature = config['arguments']['LPS_feature']
    args.post_suffix = config['arguments']['post_suffix'] 
    args.DNN_suffix = config['arguments']['DNN_suffix'] 
    args.task_challenge = config['arguments']['task_challenge'] 
    args.DNN_type = config['network']['DNN_type'] 
    args.n_hid = config['network']['n_hid']
    args.n_epochs = config['training']['n_epochs']
    args.batch_size = config['training']['batch_size']
    args.n_sources = config['training']['n_sources']
    args.frame_neigh = config['neighbour']['frame_neigh']
    args.val_samples_check = config['validation']['val_samples_check']
    args.validation_batch_size = config['validation']['validation_batch_size']


    ### more config parameters
    # audio
    args.fs = config['audio']['fs']
    args.Wlength = config['audio']['Wlength']
    args.window = config['audio']['window']
    args.window_size = args.Wlength
    args.hop_size = config['audio']['hop_size']
    args.overlap = args.Wlength*config['audio']['overlap']
    args.fft_size = args.Wlength

    # doa
    args.min_frames = config['DOA']['min_frames']
    args.n_sources = config['DOA']['n_sources']
    args.n_features = config['DOA']['n_features']
    args.n_channels = config['DOA']['n_channels']
    args.n_freq = config['DOA']['n_freq']
    args.n_doas_mix = config['DOA']['n_doas_mix']
    args.n_utt = config['DOA']['n_utt']
    args.n_mix = args.n_utt*args.n_doas_mix #22500

    # set paths
    args.STORAGE_PATH = config_env["STORAGE_PATH"]
    args.PROJECT_PATH = os.path.join(args.STORAGE_PATH, args.challenge_name)
    args.DATASET_PATH =  os.path.join(args.PROJECT_PATH, 'DATASETS')
    args.RESULTS_PATH =  os.path.join(args.PROJECT_PATH, 'RESULTS')      

    ### validation paths
    # audio
    args.audio_path = os.path.join(args.WORK_PATH, "audio")
    args.audio_challenge_path = os.path.join(args.audio_path, args.challenge_name)
    args.audio_challenge_mixtures_path = os.path.join(args.audio_challenge_path, 'mixtures')
    args.audio_challenge_clean_path = os.path.join(args.audio_challenge_path, 'clean')
    args.audio_challenge_separated_path = os.path.join(args.audio_challenge_path, 'separated')
    args.audio_challenge_separated_gt_path = os.path.join(args.audio_challenge_path, 'separated_gt')

    ### figures
    args.fig_path = os.path.join(args.WORK_PATH, "figures")
    args.fig_challenge_path = os.path.join(args.fig_path, args.challenge_name)
    args.fig_challenge_masks_path = os.path.join(args.fig_path, args.challenge_name, 'masks')
    args.fig_challenge_mixtures_path = os.path.join(args.fig_challenge_path, 'mixtures')
    args.fig_challenge_clean_path = os.path.join(args.fig_challenge_path, 'clean')
    args.fig_challenge_separated_path = os.path.join(args.fig_challenge_path, 'separated')
    args.fig_challenge_separated_gt_path = os.path.join(args.fig_challenge_path, 'separated_gt')


    ### fix paths
    if args.post_suffix != '': 
        args.post_suffix = '_' + args.post_suffix

    
    if args.DNN_suffix != '':
        args.DNN_suffix = '_' + args.DNN_suffix
    
    # number of sources
    if args.n_sources == 1:
        args.n_sources_string = 'One'
        
    elif args.n_sources == 2:
        args.n_sources_string = 'Two'
        
    elif args.n_sources == 3:
        args.n_sources_string = 'Three'

    elif args.n_sources == 4:
        args.n_sources_string = 'Four'


    ### datasets paths
    # train
    args.train1_path = os.path.join(args.STORAGE_PATH, 'datasets', 'L3DAS', 'Task1', 'L3DAS22_Task1_train360_1')
    args.train2_path = os.path.join(args.STORAGE_PATH, 'datasets', 'L3DAS', 'Task1', 'L3DAS22_Task1_train360_2')
    args.data1_train_path = os.path.join(args.train1_path, 'data')
    args.data2_train_path = os.path.join(args.train2_path, 'data')
    args.labels1_train_path = os.path.join(args.train1_path, 'labels')
    args.labels2_train_path = os.path.join(args.train2_path, 'labels')
    
    # val
    args.val_path = os.path.join(args.STORAGE_PATH, 'datasets', 'L3DAS', 'Task1', 'L3DAS22_Task1_dev')
    args.data_val_path = os.path.join(args.val_path, 'data')
    args.labels_val_path = os.path.join(args.val_path, 'labels')

    # test
    args.test_path = os.path.join(args.STORAGE_PATH, 'datasets', 'L3DAS', 'Task1', 'L3DAS22_Task1_test')
    args.data_test_path = os.path.join(args.test_path, 'data')
    args.labels_test_path = os.path.join(args.test_path, 'labels')


    ### load path
    args.load_TrainData_path = os.path.join(args.DATASET_PATH, args.case_name, 'InputData/TrainData')
    args.load_train_room_path = os.path.join(args.load_TrainData_path, 'Room' + args.Train_Room)
    args.load_train_data_path = os.path.join(args.load_train_room_path, args.n_sources_string + 'Sources' + args.post_suffix)
    
    args.load_val_data_path = os.path.join(args.load_train_room_path.replace('TrainData','ValData'), args.n_sources_string + 'Sources' + args.post_suffix)
    args.load_stats_data_path = os.path.join(args.load_train_data_path, 'stats.h5')
    args.models_path = os.path.join(args.RESULTS_PATH, args.task_challenge, 'models', args.n_sources_string + 'Sources' + args.post_suffix, args.DNN_type)
    

    ### save paths
    args.save_TrainData_path = os.path.join(args.DATASET_PATH, args.case_name, 'InputData/TrainData')
    args.save_train_room_path = os.path.join(args.save_TrainData_path, 'Room' + args.Train_Room)
    args.save_train_data_path = os.path.join(args.save_train_room_path, args.n_sources_string + 'Sources' + args.post_suffix)
    args.save_stats_data_path = os.path.join(args.save_train_data_path, 'stats.h5')

    args.save_ValData_path = os.path.join(args.DATASET_PATH, args.case_name, 'InputData/ValData')
    args.save_val_room_path = os.path.join(args.save_ValData_path, 'Room' + args.Train_Room)
    args.save_val_data_path = os.path.join(args.save_val_room_path, args.n_sources_string + 'Sources' + args.post_suffix)

    args.save_TestData_path = os.path.join(args.DATASET_PATH, args.case_name, 'InputData/TestData')
    args.save_test_room_path = os.path.join(args.save_TestData_path, 'Room' + args.Train_Room)
    args.save_test_data_path = os.path.join(args.save_test_room_path, args.n_sources_string + 'Sources' + args.post_suffix)

    return args, config