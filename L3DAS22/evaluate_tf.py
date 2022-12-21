import argparse
import os
import pickle
import sys

import numpy as np
import soundfile as sf
import torch
import torch.utils.data as utils
from tqdm import tqdm

from metrics import task1_metric
from models.FaSNet import FaSNet_origin, FaSNet_TAC
from models.MMUB import MIMO_UNet_Beamforming
from utility_functions import load_model, save_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from scipy import signal
sys.path.append('../Speech_separation/')
from generator import *
from aux import *
from prepare_inputs_individual import *
from train_networks import *


'''
Load pretrained model and compute the metrics for Task 1
of the L3DAS22 challenge. The metric is: (STOI+(1-WER))/2
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

def create_model_inputs(args, x, mean, std):

    # get channels
    p0_mix, py_mix, pz_mix, px_mix = x
    #print(x.shape, p0_mix.shape)

    # spectrograms STFT
    X = fill_spectrogram_tensor(args, p0_mix, px_mix, py_mix)

    # extract features
    theta_deg, Gx, Gy = extract_features(args, X)

    # normalization
    for f in range(0,len(theta_deg )):
        theta_deg[f,:] = (theta_deg[f,:]-mean[f])/std[f]
        Gx[f,:] = (Gx[f,:]-mean[f])/std[f]
        Gy[f,:] = (Gy[f,:]-mean[f])/std[f]

    # group features together
    features_all = [theta_deg, Gx, Gy]

    # transform into numpy
    data = np.array(features_all)

    # swap and expand axes
    x_swap = np.einsum('kli->lik', data)
    x_swap_exp = np.expand_dims(x_swap, axis=0)

    # add contextual frames
    X_in0 = neighbour1(x_swap_exp, args.frame_neigh)

    return X_in0


def enhance_sound(args, predictors, model, device, length, overlap, mean, std):
    '''
    Compute enhanced waveform using a trained model,
    applying a sliding crossfading window
    '''

    def pad(x, d):
        #zeropad to desired length
        pad = torch.zeros((x.shape[0], x.shape[1], d))
        pad[:,:,:x.shape[-1]] = x
        return pad

    def xfade(x1, x2, fade_samps, exp=1.):
        #simple linear/exponential crossfade and concatenation
        out = []
        fadein = np.arange(fade_samps) / fade_samps
        fadeout = np.arange(fade_samps, 0, -1) / fade_samps
        fade_in = fadein * exp
        fade_out = fadeout * exp
        x1[:,:,-fade_samps:] = x1[:,:,-fade_samps:] * fadeout
        x2[:,:,:fade_samps] = x2[:,:,:fade_samps] * fadein
        left = x1[:,:,:-fade_samps]
        center = x1[:,:,-fade_samps:] + x2[:,:,:fade_samps]
        end = x2[:,:,fade_samps:]
        return np.concatenate((left,center,end), axis=-1)

    overlap_len = int(length*overlap)  #in samples
    total_len = predictors.shape[-1]
    starts = np.arange(0,total_len, overlap_len)  #points to cut
    #iterate the sliding frames
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i] + length
        if end < total_len:
            cut_x = predictors[:,:,start:end]
        else:
            #zeropad the last frame
            end = total_len
            cut_x = pad(predictors[:,:,start:end], length)

        # torch to numpy
        cut_x = np.squeeze(cut_x.cpu().detach().numpy())

        # stft
        [_, _, X] = signal.stft(cut_x, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)

        #print('cut_x.shape', cut_x.shape, X.shape)

        # create inputs compatible with model format
        X_in = create_model_inputs(args, cut_x, mean, std)

        #compute model's output
        mask = model.predict(X_in)

        # get separated spectrograms
        X_separated = np.multiply(X, mask.T)

        #X_xseparated = X_separated[3,:,:]
        #X_yseparated = X_separated[1,:,:]

        _, x_separated = signal.istft(X_separated, args.fs, args.window, args.Wlength, args.overlap, args.fft_size)

        #print('x_separated.shape',x_separated.shape)
        # average x and y channels to match my paper results

        predicted_x = np.mean(np.vstack([x_separated[3,:], x_separated[1,:]]), axis=0)
        #predicted_x = model(cut_x, device) ### REPLACE MODEL OUTPUT
        #predicted_x = predicted_x.cpu().numpy()

        # expand dimensions to merge L3DAS script format
        #print('before',predicted_x.shape)
        predicted_x = np.expand_dims(np.expand_dims(predicted_x, axis=0), axis=0)
        #print('predicted_x.shape',predicted_x.shape)

        #print('after', predicted_x.shape)

        #reconstruct sound crossfading segments
        if i == 0:
            recon = predicted_x
        else:
            #print(recon.shape, predicted_x.shape)
            recon = xfade(recon, predicted_x, overlap_len)

    #undo final pad
    recon = recon[:,:,:total_len]

    return recon


def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    print ('\nLoading dataset')
    #LOAD DATASET
    #print(args.predictors_path)
    with open(args.predictors_path, 'rb') as f:
        predictors = pickle.load(f)
    with open(args.target_path, 'rb') as f:
        target = pickle.load(f)

    print ('\nPacking into numpy arrays')
    predictors = np.array(predictors)
    target = np.array(target)

    print ('\nShapes:')
    print ('Predictors: ', predictors.shape)
    print ('Target: ', target.shape)

    #convert to tensor
    predictors = torch.tensor(predictors).float()
    target = torch.tensor(target).float()
    #build dataset from tensors
    dataset_ = utils.TensorDataset(predictors, target)
    #build data loader from dataset
    dataloader = utils.DataLoader(dataset_, 1, shuffle=False, pin_memory=True)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # load config
    args, config = create_args()

    #LOAD MODEL
    # sort models by date
    models_path = '/media/alfredo/storage/L3DAS22/RESULTS/Task1/models/OneSources/MLP'
    models_list = sorted(Path(models_path).iterdir(), key=os.path.getmtime)
    
    # remove any .log file from models_list
    models_list = [i for i in models_list if '.log' not in os.path.basename(i)]

    best_model_idx = 248
    model = load_model( models_list[best_model_idx], compile=True, custom_objects={'cost': cost}) # compile=False restore optimizer state
    model.summary()

    stats_path = '/vol/vssp/mightywings/L3DAS22/DATASETS/B_format/InputData/TrainData/Room12BB01/OneSources/stats.h5'
    # load stats
    with h5py.File(stats_path, 'r') as hf:
        stats = hf.get('stats')
        stats = np.array(stats)
        mean = stats[0][0]
        std = stats[0][1]  

    #COMPUTING METRICS
    print("COMPUTING TASK 1 METRICS")
    print ('M: Final Task 1 metric')
    print ('W: Word Error Rate')
    print ('S: Stoi')

    WER = 0.
    STOI = 0.
    METRIC = 0.
    count = 0

    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            
            print(example_num)
            #print('initially',x.shape, target.shape)
            target = target.cpu().detach().numpy()
            outputs = enhance_sound(args, x, model, device, 76672, 0.5, mean, std) # recon = outputs

            #print('before',outputs.shape, target.shape)

            outputs = np.squeeze(outputs)
            target = np.squeeze(target) # edited this
            # (192000,) torch.Size([192000])

            #print('after',outputs.shape, target.shape)

            # outputs = outputs / np.max(outputs) * 0.9  #normalize prediction
            metric, wer, stoi = task1_metric(target, outputs)


            if metric is not None:

                METRIC += (1. / float(example_num + 1)) * (metric - METRIC)
                WER += (1. / float(example_num + 1)) * (wer - WER)
                STOI += (1. / float(example_num + 1)) * (stoi - STOI)

                #save sounds
                save_sounds_freq = None
                if save_sounds_freq is not None:
                    sounds_dir = os.path.join(results_path, 'sounds')
                    if not os.path.exists(sounds_dir):
                        os.makedirs(sounds_dir)

                    if count % save_sounds_freq == 0:
                        sf.write(os.path.join(sounds_dir, str(example_num)+'.wav'), outputs, 16000, 'PCM_16')
                        #print ('metric: ', metric, 'wer: ', wer, 'stoi: ', stoi)
            else:
                print ('No voice activity on this frame')
            pbar.set_description('M:' +  str(np.round(METRIC,decimals=3)) +
                   ', W:' + str(np.round(WER,decimals=3)) + ', S: ' + str(np.round(STOI,decimals=3)))
            pbar.update(1)
            count += 1

    #visualize and save results
    results = {'word error rate': WER,
               'stoi': STOI,
               'task 1 metric': METRIC
               }
    print ('*******************************')
    print ('RESULTS')
    for i in results:
        print (i, results[i])
    out_path = os.path.join(results_path, 'task1_metrics_dict.json')
    np.save(out_path, results)


if __name__ == '__main__':
    base_path = '/media/alfredo/storage/L3DAS22'
    results_path = os.path.join(base_path, 'RESULTS/Task1/metrics')
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default=os.path.join(base_path, 'RESULTS/Task1/checkpoint'))
    parser.add_argument('--results_path', type=str, default=os.path.join(base_path, 'RESULTS/Task1/metrics'))
    parser.add_argument('--save_sounds_freq', type=int, default=None)
    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default=os.path.join(base_path, 'DATASETS/processed/task1_predictors_test_uncut.pkl'))
    parser.add_argument('--target_path', type=str, default=os.path.join(base_path, 'DATASETS/processed/task1_target_test_uncut.pkl'))
    parser.add_argument('--sr', type=int, default=16000)
    #reconstruction parameters
    parser.add_argument('--segment_length', type=int, default=76672)
    parser.add_argument('--segment_overlap', type=float, default=0.5)
    #model parameters
    parser.add_argument('--architecture', type=str, default='MIMO_UNet_Beamforming',
                        help="model name")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--segment_size', type=int, default=24)
    parser.add_argument('--nspk', type=int, default=1)
    parser.add_argument('--win_len', type=int, default=16)
    parser.add_argument('--context_len', type=int, default=16)
    parser.add_argument('--fft_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=128)
    parser.add_argument('--input_channel', type=int, default=4)

    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)

    main(args)
