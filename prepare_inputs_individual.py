#python prepare_input_L3DAS.py B_format train 12BB01 12BB01 ['theta','MV'] ''
from multiprocessing.forkserver import connect_to_new_process
import os
import sys
import time
import scipy.io
from scipy.io import wavfile
from scipy import signal
from scipy.stats import vonmises
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from stats import RunningStats
import h5py
import librosa
from impulse_response import *
from create_args import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


def check_nans(X):
    array_sum = np.sum(X)
    array_has_nan = ~np.isfinite(array_sum)

    if array_has_nan == True:
        # print wrong elements
        #print(np.argwhere(~np.isfinite(X)))
        print(X[~np.isfinite(X)])
        sys.exit(-1)
    

def pad_matrix(X, max_length):

    X_list = []

    for idx in range(len(X)):

        print(idx)

        X_nparray = np.array(X[idx])
        X_padded = np.pad(X_nparray, [(0,0),(0, max_length-X_nparray.shape[1])], mode='constant', constant_values=0).tolist()
        X_list.append(X_padded)
        del X_nparray, X_padded

    return X_list


def mag(x):
   
   x = 20 * np.log10(np.abs(x))
   return x

    
def angles_dist(P0, Gx, Gy):
 
    Y = (np.conj(P0)*Gy).real
    X = (np.conj(P0)*Gx).real
    theta =  np.arctan2( Y,X )
    theta_deg = (np.degrees(theta)+360)%360 ### REMOVED .astype(int)
    
    return theta_deg
    

def evaluate_IRM(X,source_index):
    
    IRM = np.abs(X[source_index,:,:,:] ) / np.sum( np.abs(X[:,:,:,:] ),axis=0 ) 
    
    return IRM

    
def fill_spectrogram_tensor(args, p0, px, py):
    
    # STFT
    [_, _, P0] = signal.stft(p0, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    [_, _, Gx] = signal.stft(px, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    [_, _, Gy] = signal.stft(py, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    
    # fill tensor
    X = np.zeros([P0.shape[0],P0.shape[1],3], np.complex64)

    X[:,:,0] = P0
    X[:,:,1] = Gx
    X[:,:,2] = Gy

    return X


def extract_IRMs(args, a, x_clean):

    p0_mix, py_mix, pz_mix, px_mix = a

    rir_length = 0.2
    RIR = analyze_audio(a.T, x_clean, rir_length, args.fs, 3, 4) ### FIX CHANNEL ORDER
    h = RIR[0]

    # apply RIR
    x_reverb_x = signal.convolve(h[3,:], x_clean)[:len(x_clean)] # x axis
    x_reverb_y = signal.convolve(h[1,:], x_clean)[:len(x_clean)] # y axis

    # STFT
    [_, _, X_reverb_x] = signal.stft(x_reverb_x, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    [_, _, X_reverb_y] = signal.stft(x_reverb_y, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    [_, _, X_clean] = signal.stft(x_clean, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)

    # subtract reverberant speech from mixtures
    px_diff = px_mix-x_reverb_x
    py_diff = py_mix-x_reverb_y

    # STFT
    [_, _, Px_diff] = signal.stft(px_diff, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    [_, _, Py_diff] = signal.stft(py_diff, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)


    # define IRMs (Ideal Ratio Masks)
    #IRM = 0.5*(np.abs(X_clean) / (np.abs(Px_diff)+np.abs(X_clean)+1e-16) + np.abs(X_clean) / (np.abs(Py_diff)+np.abs(X_clean)+1e-16))
    IRM_speech = 0.5*( np.abs(X_reverb_x)**2 /  (np.abs(Px_diff)**2+np.abs(X_reverb_x)**2+1e-16) + np.abs(X_reverb_y)**2 / (np.abs(Py_diff)**2+np.abs(X_reverb_y)**2+1e-16))
    IRM_noise = 0.5*( np.abs(Px_diff)**2 /  (np.abs(Px_diff)**2+np.abs(X_reverb_x)**2+1e-16) + np.abs(Py_diff)**2 / (np.abs(Py_diff)**2+np.abs(X_reverb_y)**2+1e-16))  
    
    return IRM_speech, IRM_noise


def extract_features(args, X):

    # define angular feature
    theta_deg = angles_dist(X[:,:,0], X[:,:,1], X[:,:,2])

    # Define LPS
    Gx = mag(X[:,:,1]+1e-16) # add 1e-16 to avoid nans
    Gy = mag(X[:,:,2]+1e-16)

    return theta_deg, Gx, Gy

    
    
def prepare_input_mask(args):

    if args.task == 'train':
        data1 = os.listdir(args.data1_train_path)
        data1 = [args.data1_train_path + '/' + i for i in data1 if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
        data2 = os.listdir(args.data2_train_path)
        data2 = [args.data2_train_path + '/' + i  for i in data2 if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
        data = data1 + data2

    elif args.task == 'val':
        data = os.listdir(args.data_val_path)
        data = [args.data_val_path + '/' + i for i in data if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B

    elif args.task == 'test':
        data = os.listdir(args.data_test_path)
        data = [args.data_test_path + '/' + i for i in data if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
    
    # running mean and std
    if args.task == 'train':
        runstats = RunningStats(args.n_freq, np.float64)
        stats = []

    ### INPUT DATA GENERATION
    print("Filling data tensors", time.ctime())

    # initialization
    mix_index = 0

    for sound_path in data:
        sound = os.path.basename(sound_path)

        # get signals paths
        clean_path = sound_path.replace('data/','labels/').replace('_A','')
        print(mix_index, sound, sound_path)
        

        # mixtures
        a, sr = librosa.load(sound_path, sr=16000, mono=False)
        #p0_init, py_init, pz_init, px_init = a
        p0_mix, py_mix, pz_mix, px_mix = a
        
        # clean speech
        x_clean, sr_clean = librosa.load(clean_path, sr=args.fs, mono=False)

        # pressure tensors mixtures
        X = fill_spectrogram_tensor(args, p0_mix, px_mix, py_mix)

        # create IRM
        IRM = extract_IRMs(args, a, x_clean)

        # check if IRM is working
        #[_, _, Px_mix] = signal.stft(px_mix, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
        #X_separated = np.multiply(Px_mix,IRM[0])
        #X_noise = np.multiply(Px_mix,IRM[1])
        #_, x_separated = signal.istft(X_separated, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
        #_, x_noise = signal.istft(X_noise, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
        #sf.write(os.path.join(args.figures_path, "x_separated" + '_' + str(mix_index) + ".wav"), x_separated, sr)
        #sf.write(os.path.join(args.figures_path, "x_noise" + '_' + str(mix_index) + ".wav"), x_noise, sr)
        #sf.write(os.path.join(args.figures_path, "p0_mix" + '_' + str(mix_index) + ".wav"), p0_mix, sr)
                
        # create features
        theta_deg, Gx, Gy = extract_features(args, X)
        
        
        ### PLOTS
        '''
        fig,ax = plt.subplots(1,figsize=(14,14))
        n, bins, patches = plt.hist(theta_deg.flatten(), bins=72, facecolor='blue', alpha=0.5, edgecolor='black', linewidth=1.2)
        plt.tick_params(labelsize=28)
        plt.xlabel(r'$\theta$ (degrees)', fontsize=36)
        plt.ylabel(r'$\theta$ count', fontsize=36)
        plt.title('Distribution of angles', fontsize=42)
        plt.savefig(os.path.join(args.figures_path, 'theta_deg' + str(mix_index)),bbox_inches='tight')
        plt.clf()
        
        plt.imshow(theta_deg, cmap='jet', aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Distribution  of angles', fontsize=42)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(args.figures_path, 'theta_tensor' + str(mix_index)  ), bbox_inches='tight')
        plt.clf()

        
        plt.imshow(mag(X[:,:,1]), cmap='jet', aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Spectrogram: ' + '$G_x$', fontsize=42)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(args.figures_path, 'Gx' + str(mix_index)  ),bbox_inches='tight' )
        plt.clf()

        
        plt.imshow(mag(X[:,:,2]), cmap='jet', aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Spectrogram: ' + '$G_y$', fontsize=42)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(args.figures_path, 'Gy' + str(mix_index)  ),bbox_inches='tight')
        plt.clf()

        plt.imshow(IRM[0], cmap='jet', aspect='auto', interpolation='none')
        clb = plt.colorbar()
        ax.invert_yaxis()
        plt.plot()
        plt.savefig(os.path.join(args.figures_path, 'IRM' + str(mix_index)))
        plt.clf()

        plt.imshow(IRM[1], cmap='jet', aspect='auto', interpolation='none')
        clb = plt.colorbar()
        ax.invert_yaxis()
        plt.plot()
        plt.savefig(os.path.join(args.figures_path, 'IRM_noise' + str(mix_index)))
        plt.clf()
    	'''
        

        if args.task == 'train':
            # update stats
            runstats.update(np.hstack([theta_deg, Gx, Gy ]).T )

            # stats
            mean = np.zeros([args.n_freq])
            std = np.zeros([args.n_freq])
            
            mean =  runstats.stats['mean']
            std = np.sqrt(runstats.stats['var'])


        ### LOAD STATS FOR VAL
        if (args.task == 'val' or args.task == 'test' and mix_index == 0):
            # use train mean and std
            with h5py.File(args.save_stats_data_path, 'r') as hf:
                stats = hf.get('stats')
                stats = np.array(stats)
                mean = stats[0][0]
                std = stats[0][1]  
        
        # normalization
        for f in range(0,len(theta_deg )):
            theta_deg[f,:] = (theta_deg[f,:]-mean[f])/std[f]
            Gx[f,:] = (Gx[f,:]-mean[f])/std[f]
            Gy[f,:] = (Gy[f,:]-mean[f])/std[f]

        features_all = [theta_deg, Gx, Gy]

        ### FINAL CHECK
        '''
        theta_list = theta_deg.ravel()
        nbins = 72
        step=0.1
        start = np.floor(min(theta_list) / step) * step
        stop = max(theta_list) + step

        bin_edges = np.arange(start, stop, step=step)

        fig,ax = plt.subplots(1,figsize=(14,14))
        #N, bins, patches = ax.hist(data,bins=[0.3, 0.8,1.3,1.8,2.3,3.5],edgecolor='white')
        n, bins, patches = ax.hist(theta_list, bin_edges) #N, bins, patches = ax.hist(data,bins=[0.3, 0.8,1.3,1.8,2.3,3.5],edgecolor='white')
        plt.tick_params(labelsize=28)
        #   plt.xticks(bin_edges, rotation=90)
        plt.xlabel(r'$\theta$ (degrees)', fontsize=36)
        plt.ylabel(r'$\theta$ count', fontsize=36)
        plt.title('Distribution of angles', fontsize=42)
        plt.savefig(os.path.join(args.figures_path, 'test_theta_deg' + str(mix_index)),bbox_inches='tight')
        plt.clf()


        plt.imshow(theta_deg, cmap='jet', aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Distribution  of angles', fontsize=42)
        plt.savefig(os.path.join(args.figures_path, 'test_theta' + str(mix_index)), bbox_inches='tight')
        plt.clf()
        print(os.path.join(args.figures_path, 'test_theta' ))

        plt.imshow(Gx, cmap='jet', aspect='auto', interpolation='none')
        clb = plt.colorbar()
        plt.tick_params(labelsize=28)
        plt.xlabel('Time', fontsize=36)
        plt.ylabel('Frequency', fontsize=36)
        clb.ax.tick_params(labelsize=28) 
        plt.title('Distribution  of angles', fontsize=42)
        plt.savefig(os.path.join(args.figures_path, 'test_Gx' + str(mix_index)), bbox_inches='tight')
        plt.clf()
    	'''
        
        # save data
        if args.task == 'train':
            save_path = args.save_train_data_path
        
        elif args.task == 'val':
            save_path = args.save_val_data_path

        elif args.task == 'test':
            save_path = args.save_test_data_path

        with h5py.File(os.path.join(save_path, sound.replace('.wav','.h5').replace('_A','')), 'w') as hf:
            hf.create_dataset('data', data=features_all)
            hf.create_dataset('IRM', data=IRM)

        mix_index += 1
        #if mix_index == 3:
        #    break

    # save running stats the end of the loop
    if args.task == 'train':
        stats.append([mean,std])
        print(mean, std)

        # save data
        with h5py.File(args.save_stats_data_path, 'w') as hf:
            hf.create_dataset('stats', data=stats)

        
    print("Done!.", time.ctime())
   
   
if __name__ == '__main__':
    
    args, config = create_args()

    # main paths
    if args.task == 'train':
        args.figures_path = os.path.join(args.WORK_PATH, 'figures/train')
    elif args.task == 'val':
        args.figures_path = os.path.join(args.WORK_PATH, 'figures/val')
    elif args.task == 'test':
        args.figures_path = os.path.join(args.WORK_PATH, 'figures/test')

    # create path if does not exist
    if not os.path.exists(args.figures_path):
        os.makedirs(args.figures_path)

    if not os.path.exists(args.save_train_data_path):
        os.makedirs(args.save_train_data_path)
    if not os.path.exists(args.save_val_data_path):
        os.makedirs(args.save_val_data_path)
    if not os.path.exists(args.save_test_data_path):
        os.makedirs(args.save_test_data_path)    
        

    prepare_input_mask(args)
