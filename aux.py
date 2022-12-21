import os
import numpy as np
from scipy import signal
import soundfile
import matplotlib.pyplot as plt
import generator
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


# predict masks
def predict_mask(args, file_path, i):

    #with args.graph.as_default():
    set_session(args.sess)

    X, Y_gt = generator.create_batch(args, args.load_val_data_path, [file_path], args.frame_neigh)
    
    # get central frame
    Y_gt = Y_gt[:,:,args.frame_neigh,:]
    #X_fr = X[:,:,args.frame_neigh,:]

    #plot(args, os.getcwd(), X_fr[:,:,0].T, 0, mag=False)
    #plot(args, os.getcwd(), X_fr[:,:,1].T, 1, mag=False)
    #plot(args, os.getcwd(), Y_gt[:,:,0].T, 2, mag=False)

    # estimate mask        
    Y_pred = args.model.predict(X)

    # if prediction is list of masks, get the first one only for visualization purposes
    if type(Y_pred) == list:
        Y_pred = Y_pred[0]

    # plot
    plot_masks(args, args.fig_challenge_masks_path, file_path, Y_gt, Y_pred, i)

    return Y_pred, Y_gt[:,:,0]


# apply mask to mixture
def separate_speech(args, h5_folder, fig_folder, file_name, mask, i, channel=0):

    #print(mask.shape)
    file_name = file_name.replace('.h5', '_A.wav')

    # read audio
    x, sr =  soundfile.read(os.path.join(h5_folder, file_name) )
    x_ch = x[:,channel]

    # stft
    [_, _, X] = signal.stft(x_ch, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)

    # apply mask
    X_separated = np.multiply(X, mask.T)

    # plots
    plot(args, fig_folder, X_separated, i, mag=True)

    # istft
    _, x_separated = signal.istft(X_separated, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)
    
    return x_separated


### AUDIO
def write_audio(args, h5_folder, audio_folder, file_name, i, x=None, get_stft=False, channel=0):

    if 'labels' in h5_folder:
        file_name = file_name.replace('.h5', '.wav')
    
    else:
        file_name = file_name.replace('.h5', '_A.wav')

    # provide np array
    if x is not None:
        data = x
        sr = args.fs

    # read file
    else:
        data, sr =  soundfile.read(os.path.join(h5_folder, file_name) )

    # check dimensions and write to disk
    if data.ndim != 1: #type(data) is tuple:
        data = data[:,channel]

    soundfile.write(os.path.join(audio_folder, str(i) + '.wav'), data , sr)
        
    if get_stft == True:

        [_, _, X] = signal.stft(data, sr, args.window, args.Wlength, args.overlap, args.fft_size)
        
        return X
    else:

        return None


### PLOTS
# plot
def plot(args, folder, X, i, mag=True, file_name=''):

    if mag == True:
        plt.imshow(generator.mag(X+1e-16), cmap='jet', aspect='auto', interpolation='none')
    else:
        plt.imshow(X, cmap='jet', aspect='auto', interpolation='none')
    
    plt.gca().invert_yaxis()
    plt.plot()

    if file_name == '':
        plt.savefig(os.path.join(folder, str(i) + '.png'))
    else:
        plt.savefig(os.path.join(folder, file_name +  '.png'))

    plt.clf()


# plot IRM and predicted masks
def plot_masks(args, folder, file, Y, Y_pred, i):

   f, axarr = plt.subplots(2)
   axarr[0].imshow(Y[:,:,0].T, cmap='jet', aspect='auto', interpolation='none')
   axarr[1].imshow(Y_pred[:,:].T, cmap='jet', aspect='auto', interpolation='none')
   axarr[0].invert_yaxis()
   axarr[1].invert_yaxis()
   plt.plot()
   plt.savefig(os.path.join(folder, str(i) + '.png'))
   plt.clf()