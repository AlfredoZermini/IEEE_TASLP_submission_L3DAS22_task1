# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:08:14 2019

@author: xkarakonstantis
"""

#import wavfile
#from scipy import signal
import numpy as np
import math
from scipy.signal import csd, welch, cheby1, lfilter, freqz
import matplotlib.pyplot as plt
#import control as ctrl

def next_power_of_2(x):  
    return 1 if x == 0 else round(math.log(x,2))

def tfe(x, y, reg_f, **args):
   """estimate transfer function from x to y, see csd for calling convention"""   
   (F,Px) = welch(x, **args)
   (F,Py) = welch(x, **args)
   (F,Cxy) = csd(x, y, **args)
   #T_F = np.divide(Cxy,Px)
   T_F_reg = np.divide(Cxy, np.maximum(Px,reg_f))
   return (Cxy, T_F_reg)

def generate_expsweep(L,P):
    
    N= np.round(L);
    n= np.arange(N-1)
    
    x = np.sin(((L*np.pi/(2**P)))/(np.log(2**P))*np.exp((n/N)*np.log(2**P)))
    
    x = np.transpose(x)
    return x

def fconv(x, h):
    
    # FCONV Fast Convolution - matlab usage:
    #   [y] = FCONV(x, h) convolves x and h, and normalizes the output  
    #         to +-1.
    #
    #      x = input vector
    #      h = input vector
    # 
    #      See also CONV
    #
    #   NOTES:
    #
    #   1) I have a short article explaining what a convolution is.  It
    #      is available at http://stevem.us/fconv.html.
    #
    #
    #modified version of matlab code: Fast convolution, Version 1.0
    #Coded by: Stephen G. McGovern, 2003-2004.
    lx = max(np.shape(x)); lh =  max(np.shape(h))
    #Ly=max(lx,lh)
    Ly = lx + lh - 1
    Ly2=np.power(2,next_power_of_2(Ly))    # Find smallest power of 2 that is > Ly
    X=np.fft.fft(x, Ly2);		           # Fast Fourier transform
    H=np.fft.fft(h, Ly2);	               # Fast Fourier transform
    Y= X*H        	                       # 
    y=np.real(np.fft.ifft(Y, n = Ly2))     # Inverse fast Fourier transform0
    y=y[0:Ly:1]                            # Take just the first N elements
    #y=y/max(abs(y));                      # Normalize the output   
    return y
    
    
    
def analyze_audio(x_mic, x_spk, T_imp, Fs, num_Noverlap, den_noverlap):
    N_FFT = round(T_imp*Fs) #The number of data points used in each block for the FFT
    wind = np.hanning(N_FFT) #hanning window of size NFFT
    N_Overlap = round(num_Noverlap*len(wind)/den_noverlap)
    M = np.size(x_mic, axis=1) # number of mics
   # M = 4
    reg_fact = 10**(-130/10)
#    x_mic = x_mic[0:len(x_spk)]
    #m0 = 0 #channel to check out
    
    #Use more regularization at low frequencies
    reg_fact_low = 10**(-60/10) 
    # make sure input signals are column vectors
    x_spk = np.reshape(x_spk, (1,max(x_spk.shape)))
    x_mic = np.transpose(x_mic)

        
    k_F0 = round(250/(Fs/(T_imp*Fs)))
    (F,Px) = welch(x_spk, nfft = N_FFT, fs = Fs, window = wind, noverlap = N_Overlap)
    (_,Py) = welch(x_mic[0,:], nfft = N_FFT, fs = Fs, window = wind, noverlap = N_Overlap)
    (_,Cxy) = csd(x_spk, x_mic[0,:], nfft = N_FFT, fs = Fs, window = wind, noverlap = N_Overlap)


    i_max_low = np.argmax(Px[0,0:k_F0])
    reg_f = np.ones(np.size(Px))*reg_fact
    reg_f[0:i_max_low] = reg_fact_low
    (_,regH) = tfe(x_spk ,x_mic[0,:], reg_f, nfft = N_FFT, fs = Fs, window = wind, noverlap = N_Overlap)
    
    for m in range(1,M):
        (_,Traf_reg) = tfe(x_spk ,x_mic[m,:], reg_f, nfft = N_FFT, fs = Fs, window = wind, noverlap = N_Overlap)
       # H = np.column_stack((H, TraF)) 
        regH = np.vstack((regH, Traf_reg)) 
    
    temp = np.conj(np.fliplr(regH[:,1:-1]))
    tempb = np.hstack((regH,temp))
    
    Px = np.reshape(Px, np.shape(F))
    Cxy = np.reshape(Cxy, np.shape(F))

    #F1 = np.reshape(F, np.shape(regH[1,:]))
    F1 = np.reshape(F, np.shape(regH[0,:]))
    F4 = np.reshape(F, np.shape(Py))
    h_xy = np.fft.ifft(tempb)
   # h_xxyy = np.fft.ifft(H)
   
   
    # f, axes = plt.subplots(nrows= 4, ncols=1, sharex=True, sharey=True)
    # axes[0].plot(F1, ctrl.mag2db(abs(regH[1,:])))
    # axes[0].grid(True)
    # axes[0].set_title('Transfer Function mic 1')
    # axes[1].plot(F, ctrl.mag2db(abs(Px)))
    # axes[1].grid(True)
    # axes[1].set_title('Px')
    # axes[2].plot(F4, ctrl.mag2db(abs(Py)))
    # axes[2].grid(True)
    # axes[2].set_title('Py mic 1')
    # axes[3].plot(F, ctrl.mag2db(abs(Cxy)))
    # axes[3].grid(True)
    # axes[3].set_title('Cxy mic 1')
    # f.text(0.5, 0.04, 'Frequency [Hz]', ha='center')
    # f.text(0.04, 0.5,'Amplitude [dB]', va='center', rotation='vertical')
 
    

    #correction for analysis window
    eps = np.finfo(float).eps
    
    Lwin = len(wind)
    cwin = fconv(wind,wind)
    cwin_h = cwin[Lwin-1:]
    cwin_h = cwin_h/max(abs(cwin_h))
    h_IR_cor = np.divide(h_xy,(cwin_h + eps))

    
    
    return  (np.real(h_xy), np.real(h_IR_cor), F)

def plot_impulse_response(h, Fs, L, title_str):
    
    if h.shape[0] == min(h.shape):
        h = np.transpose(h)
    leg = np.zeros(min(h.shape))
    for m in range(min(h.shape)):
        leg[m] = str(int(m))
    
    
    Flow = 100
    Fhigh = 15000
    Nbp = 6
    
    B_hp, A_hp = cheby1(Nbp,1,2*Flow/Fs,btype='highpass')
    [B_lp, A_lp] = cheby1(Nbp,1,2*Fhigh/Fs, btype='lowpass');
    
    hf = lfilter(B_lp, A_lp,h)
    hf = lfilter(B_hp, A_hp,hf)
    bi = np.ones((1,L))
    bi = bi/L
    bi = np.reshape(bi,np.size(bi))
    P_h = lfilter(bi, 1, abs(hf)**2)
    
#    P_d_h = lfilter(np.ones((1,L))/L, 1, abs(hf[:,1]-hf[:,2])**2)
#    P_d_h = abs(hf[:,1]-hf[:,2])**2
    
    t_range = np.linspace(0,np.size(h,axis = 0)/Fs, np.size(h,axis =0));
    
    H1 = np.zeros((1000,min(h.shape)))
    H2 = np.zeros((100,min(h.shape)))

    n1 = 1000
    n2 = 100
    
    for m in range(min(h.shape)):
        w1, H1[:,m] = freqz(h[:,m],1,n1)
        w2, H2[:,m] = freqz(h[:,m],1,n2)

    F1 = w1/(2*np.pi)*Fs
    F2 = w2/(2*np.pi)*Fs


    f, axes = plt.subplots(nrows= 2, ncols=1, sharex=True, sharey=True)

    p1 = axes[0].plot(F1, ctrl.mag2db(abs(H1)))
    axes[0].grid(True)
    axes[0].set_title(title_str + f' Frequency Response N = {n1} points')
    axes[0].set_xlim((0,max(F1)))
    axes[0].set_ylim((-100,0))

    p2 = axes[1].plot(F2, ctrl.mag2db(abs(H2)))
    axes[1].grid(True)
    axes[1].set_xlim((0,max(F2)))
    axes[1].set_ylim((-100,0))
    axes[1].set_title(title_str + f' Frequency Response N = {n2} points')
    plt.legend(iter(p2), leg)
    plt.legend(iter(p1), leg)


    
    plt.figure(figsize=(14, 6))    
    p3 = plt.plot(t_range, 0.5*ctrl.mag2db(P_h), label = leg)
    plt.xlim((0,(np.size(h,axis = 0)/Fs)))
    #plt.ylim(-max(abs(h[:,0])),max(abs(h[:,0])))
    plt.xlabel('time - lag')
    plt.ylabel('impulse response - magnitude dB')
    plt.title(title_str + ' IRs')
    plt.legend(iter(p3), leg)

    
    
    plt.figure(figsize=(14, 6))    
    p4 = plt.plot(t_range, hf, label = leg)
    plt.xlim((0,(np.size(h,axis = 0)/Fs)))
    #plt.ylim((-130,0))
    plt.xlabel('time - lag')
    plt.ylabel('coef')
    plt.title(title_str + ' IRs')
    plt.legend(iter(p4), leg)



    