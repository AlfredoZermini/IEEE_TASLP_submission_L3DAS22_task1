import random
import h5py
import copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import traceback
import soundfile
from aux import *
from prepare_inputs_individual import mag


# load input data
def load_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        #print(hf.keys())
        
        x = hf.get('data')
        y = hf.get('IRM') #IRM    
        
        x = np.array(x)
        y = np.array(y)   

    return x, y


# create batch of data
def create_batch(args, load_path, files_list, frame_neigh):

   paths_list = [os.path.join(load_path, i) for i in files_list  ]
   batch_idx = 0

   X = []
   Y = []
   # loop over files in batch
   for file_path in paths_list:
      
      x, y = load_data(file_path)

      x_swap = np.einsum('kli->lik', x)

      # expand dimensions to match contextual frame expansion
      x_swap_exp = np.expand_dims(x_swap, axis=0)
      
      if args.n_sources == 1:
         y_exp = np.expand_dims(np.expand_dims(y, axis=0), axis=3)
      else:
         y_swap = np.einsum('kli->lik', y)
         y_exp = np.expand_dims(y_swap, axis=0)

         del y_swap
         
      del x, y, x_swap

      '''
      f, axarr = plt.subplots(3)
      feat_idx = 0
      axarr[0].imshow(x[feat_idx,:,:], cmap='jet', aspect='auto', interpolation='none')
      axarr[1].imshow(x_swap[:,:, feat_idx], cmap='jet', aspect='auto', interpolation='none')
      axarr[2].imshow(x_swap_exp[0,:,:,feat_idx], cmap='jet', aspect='auto', interpolation='none')
      axarr[0].invert_yaxis()
      axarr[1].invert_yaxis()
      axarr[2].invert_yaxis()
      plt.plot()
      plt.savefig(os.path.join(os.getcwd(), 'Test'))
      plt.clf()
      cococ

      fig,ax = plt.subplots(1,figsize=(14,14))
      n, bins, patches = plt.hist(x_swap[:,:,0].flatten(), bins=72, facecolor='blue', alpha=0.5, edgecolor='black', linewidth=1.2)
      plt.tick_params(labelsize=28)
      plt.xlabel(r'$\theta$ (degrees)', fontsize=36)
      plt.ylabel(r'$\theta$ count', fontsize=36)
      plt.title('Distribution of angles', fontsize=42)
      plt.savefig(os.path.join(os.getcwd(), 'theta_deg'),bbox_inches='tight')
      plt.clf()
      
      f, axarr = plt.subplots(2)
      axarr[0].imshow(y[:,:], cmap='jet', aspect='auto', interpolation='none')
      axarr[1].imshow(y_exp[0,:, :,0], cmap='jet', aspect='auto', interpolation='none')
      axarr[0].invert_yaxis()
      axarr[1].invert_yaxis()
      plt.plot()
      plt.savefig(os.path.join(os.getcwd(), 'IRMs'))
      plt.clf()
      '''

      # contextual frame expansion
      X_in0 = neighbour1(x_swap_exp,frame_neigh)
      Y_out0 = neighbour1(y_exp,frame_neigh)
      X.append(X_in0)
      Y.append(Y_out0)
      del x_swap_exp, y_exp

      batch_idx += 1

   X_in = np.concatenate( X, axis=0 )
   del X
   Y_out = np.concatenate(Y, axis=0 )
   del Y
   return X_in, Y_out


   # creates spectrograms with neighbour frames
def neighbour1(X, frame_neigh):
   
   [n_samples1,n_freq,n_time,n_features] = X.shape 
   n_frames = frame_neigh*2+1 

   # define tensors
   X_neigh = np.zeros([n_samples1*n_time, n_freq, n_frames, n_features ])
   X_zeros = np.zeros([n_samples1, n_freq, n_time+frame_neigh*2, n_features ])
   
   # create X_zeros, who is basically X with an empty border of 2*frame_neigh frames
   X_zeros[:,:,frame_neigh:X_zeros.shape[2]-frame_neigh,:] = X
   
   for sample in range(0,n_samples1 ):
      for frame in range(0,n_time ):
         X_neigh[sample*n_time+frame, :, :, :] = X_zeros[sample, :, frame:frame+n_frames, :]

   del X_zeros
      
   return X_neigh


# data generator
def generator(args, load_path, files_list, batch_size, dataset):

   try:
      # batch counter
      total_batch_counter = 0

      # copy list to keep track of all the audio files
      files_left = copy.deepcopy(files_list)
      
      
      # save audio and figures
      if dataset == 'val':
         
         ### print spectrograms and save audio
         for i in range(args.val_samples_check):
            
            # mixtures
            X_mix = write_audio(args, args.data_val_path, args.audio_challenge_mixtures_path, files_list[i], i, get_stft=True)
            plot(args, args.fig_challenge_mixtures_path, X_mix, i)

            # clean
            X_clean = write_audio(args, args.labels_val_path, args.audio_challenge_clean_path, files_list[i], i, get_stft=True)
            plot(args, args.fig_challenge_clean_path, X_clean, i)

            # get masks
            Y_pred, Y_gt = predict_mask(args, files_list[i], i)

            # prediction
            x_separated_gt = separate_speech(args, args.data_val_path, args.fig_challenge_separated_gt_path, files_list[i], Y_gt, i)
            write_audio(args, args.data_val_path, args.audio_challenge_separated_gt_path, files_list[i], i, x_separated_gt)

            # ground-truth
            x_separated = separate_speech(args, args.data_val_path, args.fig_challenge_separated_path, files_list[i], Y_pred, i)
            write_audio(args, args.data_val_path, args.audio_challenge_separated_path, files_list[i], i, x_separated)
            
            # plot actual waveforms
            #plt.plot(x_separated_gt)
            #plt.savefig(os.path.join(os.getcwd(),'separated_gt.png'))
            
            #plt.plot(x_separated)
            #plt.savefig(os.path.join(os.getcwd(),'separated.png'))
            
      ### loop on generator samples
      while True:

         ###print(dataset, len(files_left))

         #print('\n',len(files_left))
         if len(files_left) >= batch_size:
            
            selection_list = random.sample(files_left, k=batch_size)
            files_left = [x for x in files_left if x not in selection_list]

         # last shorter batch
         else:
            selection_list = random.sample(files_left, k=len(files_left))

         # add frame extension
         X_in, Y_out = create_batch(args, load_path, selection_list, args.frame_neigh)
         del selection_list
         
         # get central mask frame
         Y_out = Y_out[:,:,args.frame_neigh,:]

         '''
         f, axarr = plt.subplots(2)
         feat_idx = 0
         axarr[0].imshow(X_in[4,:,:,feat_idx], cmap='jet', aspect='auto', interpolation='none')
         axarr[1].imshow(Y_out[4,:], cmap='jet', aspect='auto', interpolation='none')
         axarr[0].invert_yaxis()
         axarr[1].invert_yaxis()
         plt.plot()
         plt.savefig(os.path.join(os.getcwd(), 'Check2'))
         plt.clf()
         '''

         # yield data
         if args.n_sources == 1:
            yield [X_in], Y_out[:,:,0]
         elif args.n_sources == 2:
            yield [X_in], [Y_out[:,:,0], Y_out[:,:,1]]
         elif args.n_sources == 3:
            yield [X_in], [Y_out[:,:,0], Y_out[:,:,1], Y_out[:,:,2]]
         elif args.n_sources == 4: 
            yield [X_in], [Y_out[:,:,0], Y_out[:,:,1], Y_out[:,:,2], Y_out[:,:,3]]

         del X_in, Y_out

         total_batch_counter += 1  

         # reset list if finished
         # if len(files_left) == 0: 
         if len(files_left) < batch_size:

            print('Reset list', len(files_left))
            files_left = copy.deepcopy(files_list) 
            print('Reset list after', len(files_left), len(files_list))

            # shuffle files
            if dataset == 'train':
               random.shuffle(files_left)


         #print('\nnumber of batches yielded = %d' % total_batch_counter)
   
   except Exception:
      print(traceback.format_exc())
      print('\nSomething went wrong.')
      sys.exit(1)
      #raise
      