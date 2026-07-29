from os import times

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy as sp
import scipy.stats as stats
import warnings as warn
import xarray as xr
import time
from numba import jit, njit, prange

#convrfi
import torch
#ONLY FOR VMs
torch.backends.nnpack.enabled = False

from ConvRFI  import  RFIconv,init_RFIconv

#x: input 1d data stream
#win_coeffs: window coefficients (from??)
#M: # of taps
#P: # of points
# def pfb_fir_frontend(x, win_coeffs, M, P):
#     start = time.time()
#     W = int(x.shape[0] / M / P)
#     x_p = x.reshape((W*M, P)).T
#     h_p = win_coeffs.reshape((M, P)).T
#     x_summed = np.zeros((P, M * W - M),dtype=np.complex64)
#     for t in range(0, M*W-M):
#         x_weighted = x_p[:, t:t+M] * h_p
#         x_summed[:, t] = x_weighted.sum(axis=1)
#     end = time.time()
#     print(f'PFB FIR frontend time: {end - start} seconds')
#     return x_summed.T


@njit(parallel=True, fastmath=True, cache=True)
def _pfb_fir_kernel(x_p, h_p, M, W, P):
    T   = M * W - M
    out = np.zeros((T, P), dtype=np.complex64)
    for t in prange(T):       # parallelised over time steps
        for p in range(P):
            acc = np.complex64(0.0)
            for m in range(M):
                acc += x_p[p, t + m] * h_p[p, m]
            out[t, p] = acc
    return out                      # already (T, P), no transpose needed

def pfb_fir_numba(x, win_coeffs, M, P):
    W   = int(x.shape[0] / M / P)
    # ascontiguousarray ensures cache-friendly memory layout for Numba
    x_p = np.ascontiguousarray(x.reshape(W * M, P).T)
    h_p = np.ascontiguousarray(win_coeffs.reshape(M, P).T)
    return _pfb_fir_kernel(x_p, h_p, M, W, P)

def generate_win_coeffs(M, P, window_fn="hamming"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P)
    sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

def pfb_filterbank(x, M, P, window_fn="hamming"):
    win_coeffs = generate_win_coeffs(M, P, window_fn=window_fn)
    x_fir      = pfb_fir_numba(x, win_coeffs, M, P)
    # workers=-1 uses all CPU cores for the FFT stage
    return scipy.fft.fft(x_fir, axis=1, workers=-1)



def bpsk_alt(x,e_vec,nbits,symbol_rate,wincut,fc,Ebit,fs=800e6):
    """
    nbits: number of bits to simulate
    x = the time index length/array 
    tbit = number of samples per symbol
    e_vec = the carrier waveform
    symbol_rate = symbol rate in ksps
    wincut = idk (used in firwin for cutoff)
    fc = carrier frequency
    Ebit = idk (energy per bit?)
    """
    #binary phase shift keyed
    tbit = int( fs / (symbol_rate*1e3) )
    #print('making index range...')
    x = np.arange(nbits*tbit)
    #print('making bit seq...')
    bit_seq = np.random.randint(0,2,size=(int(len(x)/tbit)+1,))
    bit_seq = (2*bit_seq)-1
    pulse = np.ones(tbit)
    #print('making symbol seq...')
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]
    
    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window=("rectangular"))
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')
    
    #apply carrier signal
    ts = 1/fs
    #print('making carrier signal...')
    e_vec = np.exp(2.j*np.pi*fc*x*ts)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    #print('modulating...')
    
    sig = sym_seq * e_vec


    return sig,sym_seq,bit_seq

def qpsk(x,nbits,symbol_rate,wincut,fc,Ebit,fs=800e6):
    #quad phase shift keyed
    """
    x = the time index length/array 
    nbits: number of bits to simulate
    tbit = number of samples per symbol
    symbol_rate = symbol rate in ksps
    wincut = idk (used in firwin for cutoff)
    fc = carrier frequency
    Ebit = idk (energy per bit?)
    fs = sampling frequency
    """
    #derived number of samples per symbol (this should go inside each rfi generator)
    tbit = int( fs / (symbol_rate*1e3) )
    
    x = np.arange(nbits*tbit)
    bit_seq = np.random.randint(1,5,size=(int(len(x)/tbit)+1,))
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]


    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='auto')


    #apply carrier signal
    ts = 1/fs
    
    #theres a faster way to optimize qpsk i think but this is fine for overnight
    arg = (2.j*np.pi*fc*x*ts) + (1.j*(np.pi/4)*(2*sym_seq-1))
    sig = np.exp(arg)
    #e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    
    #sig = e_vec

    return sig,sym_seq,bit_seq

def ask_mod(x,e_vec,nsym,tbit,wincut,symbol_rate,fc,Ebit=0.0,N0=None,fs=800e6, biases= np.array([0.5, 1.0]) ):

    #ASK but with power levels between 0 and 1 to test dc stuff
    tbit = int( fs / (symbol_rate*1e3) )
    x= np.arange(nsym*tbit)
    symsz = int((len(x)/tbit))+1
    bit_seq = np.random.randint(0,len(biases),size=symsz).astype(np.float16)
    for i in range(len(biases)):
        bit_seq[bit_seq==i] = biases[i]
    # print(tbit)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse)[:len(x)]


    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz, cutoff=wincut/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')


    #apply carrier signal
    ts = 1/fs
    e_vec = np.exp(2.j*np.pi*fc*np.arange(len(sym_seq))*ts)
    # e_vec = np.exp(2.j*np.pi * fs/f_sim * x)
    # print(len(sym_seq))
    # print(len(e_vec))
    
    sig = sym_seq * e_vec
    return sig,sym_seq,bit_seq


def vco_complex(v_in,f0,K0,ts,tbit,nbits):
    """
    Takes an input voltage and returns a variable frequency waveform
    Inputs:
        v_in : input time-indexed voltages (1D)
        f0   : quiescent frequency of oscillator
        K0   : oscillator sensitivity (Hz / V)
        ts   : sampling time (reciprocal of sample rate)
    Output:
        v_out: output time-indexed voltages
    """
    #define stuff
    x = np.tile(np.arange(tbit),nbits)
    phase = np.empty(len(v_in))
    
    #define instantaneous frequencies and phases
    freq = f0 + K0*v_in
    phase = sp.integrate.cumulative_trapezoid(freq,dx=ts,initial=0)
    for i in range(nbits):
        phase[i*tbit:(i+1)*tbit] = phase[i*tbit]
        
    #create waveform
    arg = (2.j*np.pi*x*freq*ts) + 1.j*phase
    v_out = np.exp(arg)
    
    return v_out,freq,phase

#binary freq-shift keying - switch between 2 freqs
def bfsk(nbits,symbol_rate,f0,f1,Ebit=0.0,fs=800e6, N0=None):

    tbit = int( fs / (symbol_rate*1e3) )
    
    #make bit sequence (and turn into seq of +/- 0.5's for VCO)
    bit_seq = np.random.randint(0,high=2,size=nbits)
    pulse = np.ones(tbit)
    sym_seq = np.kron(bit_seq,pulse) - 0.5
    
    #lo-pass filter
    hann = np.hanning(int(tbit*0.2))
    # sym_seq = np.convolve(sym_seq,hann,mode='same')
    # sym_seq = sym_seq/(2*np.max(sym_seq))

    fir_sz = int(0.2*tbit)
    sinc = scipy.signal.firwin(fir_sz,cutoff=0.5/fir_sz, window="rectangular")
    sym_seq = scipy.signal.convolve(sym_seq,sinc,mode='same',method='fft')
    
    ts=1/fs
    Ebit_linear = 10**(Ebit/10.0)

    #define VCO f0,K0 based on inputs
    vco_center = (f0+f1)/2
    #assuming sym_seq = 1 corresponds to voltage = 1V
    vco_sens = (f1-f0)
    
    sig,f,p = vco_complex(sym_seq, vco_center, vco_sens, ts, tbit, nbits)

    sig *= np.sqrt(Ebit_linear)

    return sig,sym_seq,f,p

def power_mask(x,n,Nchan,SKM,Nsk,M,P):
    """x: signal
    n: noise
    Nchan: number of channels to simulate (for multi-channel RFI) - centered at fc
    SKM: M of the SK
    Nsk: number of SK blocks
    M: number of taps for PFB
    P: number of points for PFB
    """
    # print('making power mask')
    out_f = np.zeros((Nsk,Nchan),dtype=np.int8)
    out_bf = np.zeros((Nsk*SKM,Nchan),dtype=np.int8)

    fb_shape = (Nsk*SKM,Nchan)
    s_shape = (Nsk,Nchan)

    
    xfb = pfb_filterbank(x, M, P)
    # print(xfb.shape)
    # plt.imshow((np.abs(xfb)**2).T, aspect='auto', origin='lower')
    # plt.title('PFB of signal')
    # plt.xlabel('Time (SK blocks)')
    # plt.ylabel('Frequency (channels)')
    # plt.colorbar(label='Power (dB)')
    # plt.show()
    nfb = pfb_filterbank(n, M, P)

    xs = np.zeros(fb_shape,dtype=np.complex64)
    xs[:-M,:] = xfb
    xs[-M:,:] = xfb[-1,:]
    xss = np.abs(xs)**2
    
    ns = np.zeros(fb_shape,dtype=np.complex64)
    ns[:-M,:] = nfb
    ns[-M:,:] = nfb[-1,:]
    nss = np.abs(ns)**2

    x_ave = np.zeros(s_shape,dtype=np.float64)
    n_ave = np.zeros(s_shape,dtype=np.float64)
    s_db = np.zeros(s_shape,dtype=np.float64)
    
    sbig_db = np.zeros(fb_shape,dtype=np.float64)
    
    this_nbig_ave = np.mean(nss)
    for i in range(Nsk*SKM):
        sbig_db[i,:] = 10*np.log10((xss[i,:])/this_nbig_ave)
        
    out_bf[sbig_db > -10] = 1

    for i in range(Nsk):
        this_x = xss[SKM*i:SKM*(i+1),:]
        this_n = nss[SKM*i:SKM*(i+1),:]
        x_ave[i,:] = np.mean(this_x,axis=0)
        n_ave[i,:] = np.mean(this_n,axis=0)
        #print(10*np.log10((x_ave[i,:])/n_ave[i,:]))
        s_db[i,:] = 10*np.log10((x_ave[i,:])/n_ave[i,:])


    out_f[s_db > -10] = 1
    return out_f.T,s_db,sbig_db,out_bf

def power_mask(x,n,Nchan,SKM,Nsk,M,P):
    """x: signal
    n: noise
    Nchan: number of channels to simulate (for multi-channel RFI) - centered at fc
    SKM: M of the SK
    Nsk: number of SK blocks
    M: number of taps for PFB
    P: number of points for PFB
    """
    # print('making power mask')
    out_f = np.zeros((Nsk,Nchan),dtype=np.int8)
    out_bf = np.zeros((Nsk*SKM,Nchan),dtype=np.int8)

    fb_shape = (Nsk*SKM,Nchan)
    s_shape = (Nsk,Nchan)

    
    xfb = pfb_filterbank(x, M, P)
    # print(xfb.shape)
    # plt.imshow((np.abs(xfb)**2).T, aspect='auto', origin='lower')
    # plt.title('PFB of signal')
    # plt.xlabel('Time (SK blocks)')
    # plt.ylabel('Frequency (channels)')
    # plt.colorbar(label='Power (dB)')
    # plt.show()
    nfb = pfb_filterbank(n, M, P)

    xs = np.zeros(fb_shape,dtype=np.complex64)
    xs[:-M,:] = xfb
    xs[-M:,:] = xfb[-1,:]
    xss = np.abs(xs)**2
    
    ns = np.zeros(fb_shape,dtype=np.complex64)
    ns[:-M,:] = nfb
    ns[-M:,:] = nfb[-1,:]
    nss = np.abs(ns)**2

    x_ave = np.zeros(s_shape,dtype=np.float64)
    n_ave = np.zeros(s_shape,dtype=np.float64)
    s_db = np.zeros(s_shape,dtype=np.float64)
    
    sbig_db = np.zeros(fb_shape,dtype=np.float64)
    
    this_nbig_ave = np.mean(nss)
    for i in range(Nsk*SKM):
        sbig_db[i,:] = 10*np.log10((xss[i,:])/this_nbig_ave)
        
    out_bf[sbig_db > -10] = 1

    for i in range(Nsk):
        this_x = xss[SKM*i:SKM*(i+1),:]
        this_n = nss[SKM*i:SKM*(i+1),:]
        x_ave[i,:] = np.mean(this_x,axis=0)
        n_ave[i,:] = np.mean(this_n,axis=0)
        #print(10*np.log10((x_ave[i,:])/n_ave[i,:]))
        s_db[i,:] = 10*np.log10((x_ave[i,:])/n_ave[i,:])


    out_f[s_db > -10] = 1
    return out_f.T,s_db,sbig_db,out_bf


def mfrg(Rmethod, nbits, symbol_rate, fc, biases = np.array([0.5,1.0]), f0 = 100e6, f1 = 75e6, Ebit = 1.0, fs = 800e6, M = 512, wincut = 1.0, SNR = 1.0, DC = 100):
    """
    Parameters:
    Rmethod: modulation method to use (e.g. 'ask', 'bpsk', etc.)
    nbits: number of bits to simulate
    symbol_rate: symbol rate in ksps
    fc: center carrier frequency
    biases: for ask_mod, the power levels to use (between 0 and 1)
    f0: for bfsk, the lower frequency
    f1: for bfsk, the higher frequency
    Ebit: energy per bit
    fs: sampling frequency
    wincut: window cut-off frequency
    SNR: signal-to-noise ratio
    DC: duty cycle (percentage of time the signal is on)

    Returns: a 2D array of shape (nchans, nbits*tbit) containing a simulated RFI signal for each channel using the chosen method
    """
    Rmethod = Rmethod.lower().strip()

    #derived number of samples per symbol (this should go inside each rfi generator)
    tbit = int( fs / (symbol_rate*1e3) )
    
    nbits = np.ceil((1.*60*128*M)/tbit)
    # #index time array
    x = np.arange(nbits*tbit)   
    # print(sigarray.shape)
    
    # pulse = np.ones(tbit)

    if Rmethod == 'ask':
        sig = ask_mod(0, np.array([]), nbits, 0, wincut, symbol_rate, fc, Ebit, None, fs, biases)[0]

    elif Rmethod == 'bpsk':
        sig = bpsk_alt(0, np.array([]), nbits, symbol_rate, wincut, fc, Ebit, fs)[0]


    elif Rmethod == 'qpsk':
        sig = qpsk(0, nbits, symbol_rate, wincut, fc, Ebit, fs)[0]
    
    elif Rmethod == 'bfsk':
        sig = bfsk(nbits, symbol_rate, f0, f1, Ebit, fs=fs)[0]

    elif Rmethod == 'noise':
        pass
    else:
        raise ValueError('Invalid Rmethod. Must be one of: ask, bpsk, qpsk, bfsk')
    
    #ramp up
    sig *= (np.arange(len(sig)))/(len(sig))

    #duty cylcle
    time = np.arange(len(sig))
    sig [time % 1000 > 10*int(DC)] = 0.0
    

    # r = sig.shape[0] % (24*16*M)
    # if r != 0:
    #     print(f"Warning: signal length {sig.shape[0]} is not a multiple of 24*16*{M}. Truncating to {sig.shape[0] - r}.")
    #     sig = sig[:(sig.shape[0] - r)] 
    
    # mult = int(sig.shape[0] / (24*16*M))

    sig = sig[:(128*M*60)]

    #generate noise
    # adc_amp = 16
    n = np.random.RandomState().normal(0,1,size=len(sig)).astype(np.int8) + 1.j*np.random.RandomState().normal(0,1,size=len(sig)).astype(np.int8)

    powerMask = power_mask(sig, n, 128, SKM=M, Nsk=60, M=24, P=128)

    sig += n
    # print(f'Generated base signal with shape {sig.shape}')
    

    pfb_sig = pfb_filterbank(sig, 24, 128)
    signal = np.zeros((M*60, 128), dtype=np.complex64)
    signal[:-24,:] = pfb_sig
    signal[-24:,:] = pfb_sig[-1,:]
    # pfb_sig = pfb_sig[:(60*M),:]

    # sigDFT = np.fft.fft(pfb_sig, axis=0)

    return signal, powerMask
     
def SKest (data, N =1 , d =1, m=256):
    """
    data: 2d Array (nchans, nsamples)
    N: number of averaged channels (default 1)
    d = shape factor (default 1)
    m: size of time bins to average over for SK estimation
    """

    bins = data.shape[1]//m
    # print(data.shape[0])
    # print(bins)
    SKarr = np.empty((data.shape[0], bins))

    for i in range(bins):
        
        S1 = np.sum(np.abs(data[:,(i*m):(i+1)*m])**1, axis=1)
        S2 = np.sum(np.abs(data[:,(i*m):(i+1)*m])**2, axis=1)
        SK = (m*N*d+1)/(m-1) * (m*S2/S1**2 - 1)
        SKarr[:, i] = SK

    #output 2D array of shape (nchans, bins) containing SK values for each bin 
    return SKarr

def ms_SKest (data, n =1 , d =1, m=256):
    
    Tbins = data.shape[1]//m
    Cbins = data.shape[0]//n
    SKarr = np.empty((Cbins, Tbins))

    for i in range(Tbins):
        
        S1 = np.sum(np.abs(data[:,(i*m):(i+1)*m])**1, axis=1)
        S2 = np.sum(np.abs(data[:,(i*m):(i+1)*m])**2, axis=1)

        for j in range(Cbins):
            S12 = np.sum(S1[j*n:(j+1)*n])
            S22 = np.sum(S2[j*n:(j+1)*n])

            SK = (m*n*d+1)/(m*n-1) * (m*n*S22/S12**2 - 1)
            SKarr[j, i] = SK
    return SKarr   

#ConvRFI implementation

#This is a slightly modified version of the example 
def conv_det(data, agg_factor = [5,5,3,5]):
  #data must be real and in float32 format
  #aggression factor is sensitivity. [Vertical RFI, Horizontal RFI, left/right edges, top/bottom edges]

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if device!='cpu':
      torch.cuda.empty_cache()

  net = RFIconv(device=device)

  net = init_RFIconv(net,aggressive_factor = agg_factor,device=device).to(device)
  with torch.no_grad():
    output = net(torch.tensor(data.squeeze()[None,None,:,:]).to(device)).squeeze().cpu().numpy()
  data_test_ma = np.ma.masked_where((output),data.squeeze())

  if device!='cpu':
    torch.cuda.empty_cache()

  return data_test_ma 

import aoflagger

def aoflaggerMit(signal, count=50):
    nch = signal.shape[0]
    ntimes = signal.shape[1]

    flagger = aoflagger.AOFlagger()
    path = flagger.find_strategy_file(aoflagger.TelescopeId.Generic)
    strategy = flagger.load_strategy_file(path)
    data = flagger.make_image_set(ntimes, nch, 8)

    ratiosum = 0.0
    ratiosumsq = 0.0
    for repeat in range(count):
        for imgindex in range(8):

            data.set_image_buffer(imgindex, signal.astype(np.float32))

        flags = strategy.run(data)
        flagvalues = flags.get_buffer()
        ratio = float(sum(sum(flagvalues))) / (nch*ntimes)
        ratiosum += ratio
        ratiosumsq += ratio*ratio
    
    mitigated = signal.copy()
    mitigated[flagvalues==1] = 0.0
    return mitigated

def  SK_thresholds (m, n=1, d=1, p = 0.00013499):
    """
    m: Size of time bins to average over for SK estimation 
    n: number of averaged channels
    d: shape factor (default 1)
    p: desired probability of false alarm (default 0.00013499 for 5 sigma)

    returns: lower and upper SK thresholds for given parameters
    """
    x = np.random.RandomState().normal(0,1,size=m*4800).astype(np.int8) + 1.j*np.random.RandomState().normal(0,1,size=m*4800).astype(np.int8)
    xpfb = pfb_filterbank(x, 24, 128)
    # xdft = np.fft.fft(xpfb, axis=0)
    xdft = xpfb.T
    SKx = SKest(np.abs(xdft)**2, N=n, d=d,m=m)
    # print(SKx.shape)
    # skew = np.mean(SKx)*m / ((m-1)*(m-2)*np.std(SKx)**3)

    realX = SKx.real 
    realX = realX.reshape(-1)
    # print(realX.shape)
    # imagX= SKx.imag

    skewR, locR, scaleR = scipy.stats.pearson3.fit(realX)
    # skewI, locI, scaleI = scipy.stats.pearson3.fit(imagX)

    lower_threshR = scipy.stats.pearson3.ppf(p, skewR, locR, scaleR)
    upper_threshR = scipy.stats.pearson3.ppf(1 - p, skewR, locR, scaleR)
    # lower_threshI = scipy.stats.pearson3.ppf(p, skewI, locI, scaleI)
    # upper_threshI = scipy.stats.pearson3.ppf(1 - p, skewI, locI, scaleI)

    return lower_threshR, upper_threshR

def ms_SK_thresholds (m, n=1, d=1, p = 0.00013499):
    """
    m: Size of time bins to average over for SK estimation 
    n: number of averaged channels
    d: shape factor (default 1)
    p: desired probability of false alarm (default 0.00013499 for 5 sigma)
    """
    x = np.random.RandomState().normal(0,1,size=m*4800).astype(np.int8) + 1.j*np.random.RandomState().normal(0,1,size=m*4800).astype(np.int8)
    xpfb = pfb_filterbank(x, 24, 128)
    # xdft = np.fft.fft(xpfb, axis=0)
    xdft = xpfb.T
    SKx = ms_SKest(np.abs(xdft)**2, n=n, d=d,m=m)
    # print(SKx.shape)
    # skew = np.mean(SKx)*m / ((m-1)*(m-2)*np.std(SKx)**3)

    realX = SKx.real 
    realX = realX.reshape(-1)
    # print(realX.shape)
    # imagX= SKx.imag[:, 0]

    skewR, locR, scaleR = scipy.stats.pearson3.fit(realX)
    # skewI, locI, scaleI = scipy.stats.pearson3.fit(imagX)

    lower_threshR = scipy.stats.pearson3.ppf(p, skewR, locR, scaleR)
    upper_threshR = scipy.stats.pearson3.ppf(1 - p, skewR, locR, scaleR)
    # lower_threshI = scipy.stats.pearson3.ppf(p, skewI, locI, scaleI)
    # upper_threshI = scipy.stats.pearson3.ppf(1 - p, skewI, locI, scaleI)
    return lower_threshR, upper_threshR

#Defining high level functions to generate signals and test mitigation methods
def sigGen (Rmethod, nbits, symbol_rate, fc, biases = np.array([0.5,1.0]), f0 = 100e6, f1 = 75e6, Ebit = 1.0, fs = 100e6, wincut = 39, M = 16, SNR=1.0,DC=100):
    """
    Parameters:
    Rmethod: modulation method to use (e.g. 'ask', 'bpsk', etc.)
    nbits: number of bits to simulate
    symbol_rate: symbol rate in ksps
    fc: center carrier frequency
    biases: for ask_mod, the power levels to use (between 0 and 1)
    f0: for bfsk, the lower frequency
    f1: for bfsk, the higher frequency
    Ebit: energy per bit
    fs: sampling frequency
    wincut: window cut-off frequency
    SNR: signal-to-noise ratio
    DC: duty cycle (percentage of time the signal is on)
    """
    Sig, powerMask = mfrg(Rmethod, nbits, symbol_rate, fc, biases=biases, f1=f1, f0=f0, wincut=wincut, Ebit=Ebit, fs=fs, M=M, SNR=SNR, DC=DC)
    Sig = Sig.T

    #pseudo decibel conversion
    Sig_lin = np.abs(Sig)**2
    Sig_db = 10*np.log10(Sig_lin)    
    
    return Sig,Sig_lin,Sig_db, powerMask

def SKmitigate (Signal, n=1, d=1, m=256):
    """
    Signal: 2D array of shape (nchans, nsamples) containing the signal to mitigate
    n: number of averaged channels
    d: shape factor (default 1)
    m: size of time bins to average over for SK estimation

    Returns: a 2D array of the same shape as Signal with RFI mitigated using spectral kurtosis
    """
        #make sure M is compatible with the temporal length of the data
    time = Signal.shape[1]
    if time % m != 0:
                
        print(f'M (time bin size) must be a factor of the number of time samples. Got M={m} and time samples={time}.')
        if m > time:
            raise ValueError('M is larger than the number of time samples. Thats not enough samples')
        Signal = Signal[:,:-(time % m)]
        print(f'Adjusting signal length to {Signal.shape[1]} for compatibility.')
    # print(f'M: {m}')
    
    SKvals = SKest(Signal, N=n, d=d, m=m)
    lower, upper = SK_thresholds(m, n, d)
    mask = np.kron(SKvals, np.ones(m))
    mitigated = Signal.copy()
    mitigated[(mask < lower)|(mask > upper)] = 0

    return mitigated

def ms_SKmitigate (Signal, n=1, d=1, m=256):
    """
    Multi-scale spectral kurtosis mitigation. Applies SK mitigation at multiple time scales and combines masks.

    Signal: 2D array of shape (nchans, nsamples) containing the signal to mitigate
    n: number of averaged channels for ms-SK (default 1)
    d: shape factor (default 1)
    m: size of time bins to average over for SK estimation

    Returns: a 2D array of the same shape as Signal with RFI mitigated using multi-scale spectral kurtosis
"""
    #make sure M is compatible with the temporal length of the data
    time = Signal.shape[1]
    if time % m != 0:
            
        print(f'M (time bin size) must be a factor of the number of time samples. Got M={m} and time samples={time}.')
        if m > time:
            m = time
        while time % m != 0:
            m += 1
        print(f'Adjusting M to {m} for compatibility.')
    # print(f'M: {m}')

    #make sure N is compatible with the number of channels
    chans = Signal.shape[0]
    if chans % n != 0:
        print(f'N (number of averaged channels) must be a factor of the number of channels. Got N={n} and channels={chans}.')
        if n > chans:
            n = chans
        while chans % n != 0:
            n += 1
        print(f'Adjusting N to {n} for compatibility.')

    mitigated = Signal.copy()
    

    SKarr = ms_SKest(Signal, n=n, d=d, m=m)
    #output 2D array of shape (nchans, bins) containing SK values for each bin 
    mask = np.kron(SKarr, np.ones(m))
    mask = np.repeat(mask, n, axis=0)[:Signal.shape[0], :Signal.shape[1]]
    lower, upper = ms_SK_thresholds(m, n, d)
    # lower += -3
    # upper += -3
    # print(f'ms-SK thresholds: lower={lower}, upper={upper}')
    mitigated[(mask < lower)|(mask > upper)] = 0


    return mitigated, SKarr

def ConvRFI_mitigate (Signal, agg_factor = [5,5,3,5], bins = 48):
    """Signal: 2D array of shape (nchans, nsamples) containing the signal to mitigate 
    agg_factor: aggression factor for ConvRFI. [Vertical RFI, Horizontal RFI, left/right edges, top/bottom edges]
    Returns: a 2D array of the same shape as Signal with RFI mitigated using ConvRFI
    """
    # bins = 50
    bins = int(bins)
    time = Signal.shape[1]
    if time % bins != 0:
            
        print(f'Bins (time bin size) must be a factor of the number of time samples. Got bins={bins} and time samples={time}.')
        if bins > time:
            bins = time
        while time % bins != 0:
            bins += 1
        print(f'Adjusting Bins to {bins} for compatibility.')
    
    TAsig = np.empty((Signal.shape[0], bins))
    for i in range(bins):
        TAsig[:,i] = np.mean(Signal[:, i*bins:(i+1)*bins], axis=1)

    mitigated = conv_det(np.float32(np.abs(TAsig)**2), agg_factor=agg_factor)
    mitigated = np.kron(mitigated, np.ones(Signal.shape[1]//bins))
    output = Signal.copy()
    output[mitigated.mask] = 0
    return output

def errorCalculator(powermask, mitigated):
    mask = powermask[3].T
    # print(mask.shape)
    RFI = Sig_lin.copy()
    RFI[mask==1] = 0
    errorCalc = mitigated.copy()
    errorCalc[errorCalc!=0] = 1
    RFIcount = np.count_nonzero(RFI==0)
    flagged = np.count_nonzero((RFI==0) & (errorCalc==0))/RFIcount
    # print(f'Flagged RFI percentage: {flagged}')
    FPrate = np.count_nonzero((errorCalc == 0) & (mask==0)) / (np.prod(errorCalc.shape)-RFIcount)
    # print(f'False positive rate: {FPrate}')
    FN = 1-flagged
    TP = flagged
    FP = FPrate
    TN = 1-FPrate
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    # print (f'Precision: {np.log10(TP/(FP))+TP}, Accuracy Score: {TP/FP}')
    return TP,FP,np.log10(TP/(FP))+TP,TP/FP

def VoltGen(Rmethod, nbits, symbol_rate, fc, biases = np.array([0.5,1.0]), f0 = 100e6, f1 = 75e6, Ebit = 1.0, fs = 800e6, wincut = 0.15):
    """
    Parameters:
    Rmethod: modulation method to use (e.g. 'ask', 'bpsk', etc.)
    nbits: number of bits to simulate
    symbol_rate: symbol rate in ksps
    fc: center carrier frequency
    biases: for ask_mod, the power levels to use (between 0 and 1)
    f0: for bfsk, the lower frequency
    f1: for bfsk, the higher frequency
    Ebit: energy per bit
    fs: sampling frequency
    wincut: window cut-off frequency
    SNR: signal-to-noise ratio
    DC: duty cycle (percentage of time the signal is on)

    Returns: a 2D array of shape (nchans, nbits*tbit) containing a simulated RFI signal for each channel using the chosen method
    """
    Rmethod = Rmethod.lower().strip()

    #derived number of samples per symbol (this should go inside each rfi generator)
    tbit = int( fs / (symbol_rate*1e3) )
    
    nbits = np.ceil((1.*60*128*4096)/tbit)
    # #index time array
    x = np.arange(nbits*tbit)   
    # print(sigarray.shape)
    
    # pulse = np.ones(tbit)

    if Rmethod == 'ask':
        sig = ask_mod(0, np.array([]), nbits, 0, wincut, symbol_rate, fc, Ebit, None, fs, biases)[0]

    elif Rmethod == 'bpsk':
        sig = bpsk_alt(0, np.array([]), nbits, symbol_rate, wincut, fc, Ebit, fs)[0]


    elif Rmethod == 'qpsk':
        sig = qpsk(0, nbits, symbol_rate, wincut, fc, Ebit, fs)[0]
    
    elif Rmethod == 'bfsk':
        sig = bfsk(nbits, symbol_rate, f0, f1, Ebit, fs=fs)[0]

    elif Rmethod == 'noise':
        pass
    else:
        raise ValueError('Invalid Rmethod. Must be one of: ask, bpsk, qpsk, bfsk')
    
    #ramp up

    return sig 

def DuCyc (sig, DC = 100):
    #duty cylcle
    duty = np.ones(500)

    duty[int(DC*5):] = 0
    duty = np.tile(duty, len(sig)//500 + 1)[:len(sig)] 
    return sig * duty

def VolttoSig(Volt, Noise, DC = 100, M = 512, SNR= 1.0):
    sig = Volt.copy()
    sig = sig[:(128*M*60)]

    n = Noise.copy()
    n = n[:(128*M*60)]
    
    sig *= (np.arange(len(sig)))/(len(sig))

    # #duty cylcle
    # time = np.arange(len(sig))
    # sig [time % 1000 > 10*int(DC)] = 0.0
    

    # r = sig.shape[0] % (24*16*M)
    # if r != 0:
    #     print(f"Warning: signal length {sig.shape[0]} is not a multiple of 24*16*{M}. Truncating to {sig.shape[0] - r}.")
    #     sig = sig[:(sig.shape[0] - r)] 
    
    # mult = int(sig.shape[0] / (24*16*M))


    #generate noise
    # adc_amp = 16

    powerMask = power_mask(sig, n, 128, SKM=M, Nsk=60, M=24, P=128)

    sig += n
    # print(f'Generated base signal with shape {sig.shape}')
    

    pfb_sig = pfb_filterbank(sig, 24, 128)
    signal = np.zeros((M*60, 128), dtype=np.complex64)
    signal[:-24,:] = pfb_sig
    signal[-24:,:] = pfb_sig[-1,:]
    # pfb_sig = pfb_sig[:(60*M),:]

    # sigDFT = np.fft.fft(pfb_sig, axis=0)

    Sig = signal.T

    #pseudo decibel conversion
    Sig_lin = np.abs(Sig)**2
    Sig_db = 10*np.log10(Sig_lin)    
    
    return Sig,Sig_lin,Sig_db, powerMask

###Calcing all SK and ms-SK thresholds for all M and N values
# Ms = np.power(2, np.arange(7, 13))
# msSKThresh = np.empty((6,4,2))
# for i, m in enumerate(Ms):
#     for j, n in enumerate([1,2,4,8]):
#         lower, upper = ms_SK_thresholds(m, n=n, d=1)
#         msSKThresh[i,j,0] = lower
#         msSKThresh[i,j,1] = upper
    
# ### BPSK FOR SK
# SymRt = [1, 4,20,50,100,200]
# FC = [0,1/8,1/4,1/2]
# FS = [100,300,500,650,800]
# SNR = [0.5, 1, 2, 4]
# DC = [15,30,50,60,75,100]
# emptyData = np.full((len(SymRt), len(FC), len(FS), 6, len(SNR), len(DC), 5), np.nan)
# CoordsDict = {
#     # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
#     'SymbolRate': SymRt,
#     'FC': FC,
#     'FS': FS,
#     # 'Wincut': np.arange(1, 6)*0.05,
#     # 'm': m,
#     'M': np.power(2, np.arange(7, 13)),
#     # 'N': np.arange(1, 6),
#     'SNR': SNR,
#     'DC': DC,
#     'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
# }
# SKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

# ### BPSK FOR msSK
# emptyData = np.full((6, 4, 5, 3, 6,4,6, 5), np.nan)
# SymRt = [1, 4,20,50,100,200]
# FC = [0,1/8,1/4,1/2]
# FS = [100,300,500,650,800]
# n = [2,4,8]
# SNR = [0.5, 1, 2, 4]
# DC = [15,30,50,60,75,100]

# CoordsDict = {
#     # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
#     'SymbolRate': SymRt,
#     'FC': FC,
#     'FS': FS,
#     # 'Wincut': np.arange(1, 6)*0.05,
#     # 'm': m,
#     'n': n,
#     'M': np.power(2, np.arange(7, 13)),
#     # 'N': np.arange(1, 6),
#     'SNR': SNR,
#     'DC': DC,
    
#     'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
# }
# msSKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

# ###BPSK for ConvRFI
# SymRt = [1, 4,20,50,100,200]
# FC = [0,1/8,1/4,1/2]
# FS = [100,300,500,650,800]
# Agfac1 = [0.00, 0.45, 1.66, 3.00]
# Agfac2 = [0.00, 0.45, 1.66, 3.00]
# Agfac3 = [0.00, 0.45, 1.66, 3.00]
# Agfac4 = [0.00, 0.45, 1.66, 3.00]
# bins = [26, 80, 128, 160]
# SNR = [0.5, 1, 2, 4]
# DC = [15,30,50,60,75,100]
# emptyData = np.full((len(SymRt), len(FC), len(FS), len(Agfac1), len(Agfac2), len(Agfac3), len(Agfac4), len(bins), len(SNR), len(DC), 5), np.nan)

# CoordsDict = {
#     'SymbolRate': SymRt,
#     'FC': FC,
#     'FS': FS,
#     'AggressionFactor1': Agfac1,
#     'AggressionFactor2': Agfac2,
#     'AggressionFactor3': Agfac3,
#     'AggressionFactor4': Agfac4,
#     'Bins': bins,
#     'SNR': SNR,
#     'DC': DC,
#     'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
# }
# ConvRFIds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

# ###BPSK for AOFlagger
# SymRt = [1, 4,20,50,100,200]
# FC = [0,1/8,1/4,1/2]
# FS = [100,300,500,650,800]
# Count = [1, 5]
# SNR = [0.5, 1, 2, 4]
# DC = [15,30,50,60,75,100]
# emptyData = np.full((len(SymRt), len(FC), len(FS), len(Count), len(SNR), len(DC), 5), np.nan)

# CoordsDict = {
#     'SymbolRate': SymRt,
#     'FC': FC,
#     'FS': FS,
#     'Count': Count,
#     'SNR': SNR,
#     'DC': DC,
#     'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
# }
# AOFlaggerds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

# ### BPSK
# try:
#     for sr in range(1,  6):
        
#         print(f'{(sr/6)} percent complete')
        
#         for fc in range(4): 
#             for fs in range(5):  
#             # if 5*sr*30*1e6 > fs*200*1e6:
#             #     print(f'Skipping invalid configuration: SymbolRate={SymRt[sr]} ksps, FS={FS[fs]} MHz')
#             #     continue
#                 try:
#                     RawVolt = VoltGen('bpsk', 75, SymRt[sr], ((FC[fc]*(FS[fs]/128))+120)*1e6, biases= np.array([0.5, 1]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6)
#                 except Exception as e:
#                     print(f'Error generating voltage signal for SymbolRate={SymRt[sr]}, fc={FC[fc]}, FS={FS[fs]}: {e}')
#                     continue
#                 # if 24*16*128*m > 6000*nb*fs*200*1e6/(SymRt[sr]*30*1e6):
#                     #     print(f'Skipping invalid configuration: M={128*m}, nbits={6000*nb}, fs={FS[fs]} MHz, SymbolRate={SymRt[sr]} ksps')
#                     #     continue
#                 for snr in range(4):
#                     noise = np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8) + 1.j*np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8)
#                     for dc in range(6):
#                         DuVolt = DuCyc(RawVolt, DC=DC[dc])
#                         for m in range(7, 13):
#                             print(f'Processing: SymbolRate={SymRt[sr]} ksps, fc={(FC[fc]*(FS[fs]/128)+120)*1e6} Hz, FS={FS[fs]*1e6} Hz, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
#                             Sig, Sig_lin, Sig_db, powerMask = VolttoSig(DuVolt, noise, M=2**m, SNR=SNR[snr], DC=DC[dc])

#                             # Sig, Sig_lin, Sig_db, powerMask = sigGen('bpsk', 75, SymRt[sr], ((FC[fc]*(FS[fs]/128))+120)*1e6, biases= np.array([0.5, 1]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6, M=2**m, SNR=SNR[snr], DC=DC[dc])

#                             # except Exception as e:
#                             #     print (f'Error converting voltage to signal for SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}: {e}')
#                             #     continue

#                             #Adjusting Signal length to be compatible with the m value
#                             times = Sig.shape[1]
#                             if times % (2**m) != 0:                                            
#                                 print(f'M (time bin size) must be a factor of the number of time samples. Got M={2**m} and time samples={times}.')
#                                 if 2**m > times:
#                                     raise ValueError('M is larger than the number of time samples. Thats not enough samples')
#                                 Sig = Sig[:,:-(times % (2**m))]
#                                 print(f'Adjusting signal length to {Sig.shape[1]} for compatibility.')
#                             Sig_lin = Sig_lin[:,:Sig.shape[1]]
#                             Sig_db = Sig_db[:,:Sig.shape[1]]
#                             powerMask = list(powerMask)
#                             powerMask[3] = powerMask[3][:Sig.shape[1],:]
#                             start = time.time()
#                             SKmit = SKmitigate(Sig_lin, n=1, d=1, m=2**m)
                            
#                             end = time.time()
#                             SK_time = end - start
#                             try: 
#                                 SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
#                             except Exception as e:
#                                 print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
#                                 continue
#                             SKds[sr,fc,fs,m-7,snr,dc,:] = [SK_Pr, SK_Acc, SK_TP, SK_FP, SK_time]

#                             # adjust the Signal length to be compatible with the m 
#                             # Sig = Sig[:,:SKmit.shape[1]]
                            
#                             # powerMask = list(powerMask)
#                             # powerMask[3] = powerMask[3][:SKmit.shape[1],:]

#                             #Skipping m values that are not the first m value because they are redundant for the other methods
                            
                            
#                             for n_val in range(3):
#                                 # lower, upper = msSKThresh[m-7,n_val,:]
#                                 start = time.time()
#                                 ms_SKmit, SKarr = ms_SKmitigate(Sig_lin, n=n[n_val], d=1, m=2**m)
#                                 end = time.time()
#                                 msSK_time = end - start
#                                 try:
#                                     msSK_TP, msSK_FP, msSK_Pr, msSK_Acc = errorCalculator(powerMask, ms_SKmit)
#                                 except Exception as e:
#                                     print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
#                                     continue
#                                 msSKds[sr,fc,fs,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP, msSK_time]

#                             if m != 7:
#                                 continue
#                             print("I am doing convRFI and AOFlagger now: m = ", m)

#                             for A1i, A1 in enumerate(Agfac1):
#                                 for A2i, A2 in enumerate(Agfac2):
#                                     for A3i, A3 in enumerate(Agfac3):
#                                         for A4i, A4 in enumerate(Agfac4):
#                                             for bi, b in enumerate(bins):
#                                                 try:
#                                                     start = time.time()
#                                                     ConvMit = ConvRFI_mitigate(Sig_lin, agg_factor=[A1,A2,A3,A4], bins = b)
#                                                     end = time.time()
#                                                     Conv_time = end - start
#                                                 except Exception as e:
#                                                     print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
#                                                     continue
#                                                 try:
#                                                     Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
#                                                 except Exception as e:
#                                                     print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
#                                                     continue
#                                                 ConvRFIds[sr,fc,fs,A1i,A2i,A3i,A4i,bi,snr,dc,:] = [Conv_Pr, Conv_Acc, Conv_TP, Conv_FP, Conv_time] 
                                
#                             for ci, c in enumerate(Count):
#                                 try:
#                                     start = time.time()
#                                     AoMit = aoflaggerMit(Sig_lin, count=c)
#                                     end = time.time()
#                                     AO_time = end - start

#                                 except Exception as e:
#                                     print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
#                                     continue
#                                 try:
#                                     Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
#                                     pass
#                                 except Exception as e:
#                                     print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
#                                     continue
#                                 AOFlaggerds[sr,fc,fs,ci,snr,dc,:] = [Ao_Pr, Ao_Acc, Ao_TP, Ao_FP, AO_time]
                            

#                         # try:
#                         #     # SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
#                         #     # Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
#                         #     # Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
#                         # except Exception as e:
#                         #     print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
#                         #     continue
#                         # msSKds[sr,fc,fs,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP]
    
#                 print (f'Completed metrics for SymbolRate={(SymRt[sr])}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
#                 SKds.to_netcdf(f'home/scratch/amuthiya/BPSK_SK_{sr}_{fc}_{fs}.nc')
#                 msSKds.to_netcdf(f'home/scratch/amuthiya/BPSK_msSK_{sr}_{fc}_{fs}.nc')
#                 ConvRFIds.to_netcdf(f'home/scratch/amuthiya/BPSK_ConvRFI_{sr}_{fc}_{fs}.nc')
#                 AOFlaggerds.to_netcdf(f'home/scratch/amuthiya/BPSK_AOFlagger_{sr}_{fc}_{fs}.nc')
#     SKds.to_netcdf('home/scratch/amuthiya/BPSK_SK_full.nc')           
#     msSKds.to_netcdf('home/scratch/amuthiya/BPSK_msSK_full.nc')
#     ConvRFIds.to_netcdf('home/scratch/amuthiya/BPSK_ConvRFI_full.nc')
#     AOFlaggerds.to_netcdf('home/scratch/amuthiya/BPSK_AOFlagger_full.nc')
    
# except Exception as e:
#     print(f'Error in main loop: {e}')
#     SKds.to_netcdf('home/scratch/amuthiya/BPSK_SK_partial.nc')
#     msSKds.to_netcdf('home/scratch/amuthiya/BPSK_msSK_partial.nc')
#     ConvRFIds.to_netcdf('home/scratch/amuthiya/BPSK_ConvRFI_partial.nc')
#     AOFlaggerds.to_netcdf('home/scratch/amuthiya/BPSK_AOFlagger_partial.nc')


    
### QPSK FOR SK
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(FC), len(FS), 6, len(SNR), len(DC), 5), np.nan)
CoordsDict = {
    # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    # 'Wincut': np.arange(1, 6)*0.05,
    # 'm': m,
    'M': np.power(2, np.arange(7, 13)),
    # 'N': np.arange(1, 6),
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
SKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

### QPSK FOR msSK
emptyData = np.full((6, 4, 5, 3, 6,4,6, 5), np.nan)
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
n = [2,4,8]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]

CoordsDict = {
    # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    # 'Wincut': np.arange(1, 6)*0.05,
    # 'm': m,
    'n': n,
    'M': np.power(2, np.arange(7, 13)),
    # 'N': np.arange(1, 6),
    'SNR': SNR,
    'DC': DC,
    
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
msSKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

###QPSK for ConvRFI
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
Agfac1 = [0.00, 0.45, 1.66, 3.00]
Agfac2 = [0.00, 0.45, 1.66, 3.00]
Agfac3 = [0.00, 0.45, 1.66, 3.00]
Agfac4 = [0.00, 0.45, 1.66, 3.00]
bins = [26, 80, 128, 160]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(FC), len(FS), len(Agfac1), len(Agfac2), len(Agfac3), len(Agfac4), len(bins), len(SNR), len(DC), 5), np.nan)

CoordsDict = {
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    'AggressionFactor1': Agfac1,
    'AggressionFactor2': Agfac2,
    'AggressionFactor3': Agfac3,
    'AggressionFactor4': Agfac4,
    'Bins': bins,
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
ConvRFIds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

###QPSK for AOFlagger
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
Count = [1, 5]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(FC), len(FS), len(Count), len(SNR), len(DC), 5), np.nan)

CoordsDict = {
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    'Count': Count,
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
AOFlaggerds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

### QPSK
try:
    for sr in range(6):
        
        print(f'{(sr/6)} percent complete')
        
        for fc in range(4): 
            for fs in range(5):  
            # if 5*sr*30*1e6 > fs*200*1e6:
            #     print(f'Skipping invalid configuration: SymbolRate={SymRt[sr]} ksps, FS={FS[fs]} MHz')
            #     continue
                try:
                    RawVolt = VoltGen('qpsk', 75, SymRt[sr], ((FC[fc]*(FS[fs]/128))+120)*1e6, biases= np.array([0.5, 1]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6)
                except Exception as e:
                    print(f'Error generating voltage signal for SymbolRate={SymRt[sr]}, fc={FC[fc]}, FS={FS[fs]}: {e}')
                    continue
                # if 24*16*128*m > 6000*nb*fs*200*1e6/(SymRt[sr]*30*1e6):
                    #     print(f'Skipping invalid configuration: M={128*m}, nbits={6000*nb}, fs={FS[fs]} MHz, SymbolRate={SymRt[sr]} ksps')
                    #     continue
                for snr in range(4):
                    noise = np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8) + 1.j*np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8)
                    for dc in range(6):
                        DuVolt = DuCyc(RawVolt, DC=DC[dc])
                        for m in range(7, 13):
                            print(f'Processing: SymbolRate={SymRt[sr]} ksps, fc={(FC[fc]*(FS[fs]/128)+120)*1e6} Hz, FS={FS[fs]*1e6} Hz, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
                            Sig, Sig_lin, Sig_db, powerMask = VoltStoSig(DuVolt, noise, M=2**m, SNR=SNR[snr], DC=DC[dc])

                            # Sig, Sig_lin, Sig_db, powerMask = sigGen('qpsk', 75, SymRt[sr], ((FC[fc]*(FS[fs]/128))+120)*1e6, biases= np.array([0.5, 1]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6, M=2**m, SNR=SNR[snr], DC=DC[dc])

                            # except Exception as e:
                            #     print (f'Error converting voltage to signal for SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}: {e}')
                            #     continue

                            #Adjusting Signal length to be compatible with the m value
                            times = Sig.shape[1]
                            if times % (2**m) != 0:                                            
                                print(f'M (time bin size) must be a factor of the number of time samples. Got M={2**m} and time samples={times}.')
                                if 2**m > times:
                                    raise ValueError('M is larger than the number of time samples. Thats not enough samples')
                                Sig = Sig[:,:-(times % (2**m))]
                                print(f'Adjusting signal length to {Sig.shape[1]} for compatibility.')
                            Sig_lin = Sig_lin[:,:Sig.shape[1]]
                            Sig_db = Sig_db[:,:Sig.shape[1]]
                            powerMask = list(powerMask)
                            powerMask[3] = powerMask[3][:Sig.shape[1],:]
                            
                            start = time.time()
                            SKmit = SKmitigate(Sig_lin, n=1, d=1, m=2**m)
                            end = time.time()
                            SK_time = end - start
                            try: 
                                SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
                            except Exception as e:
                                print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                continue
                            SKds[sr,fc,fs,m-7,snr,dc,:] = [SK_Pr, SK_Acc, SK_TP, SK_FP, SK_time]

                            # adjust the Signal length to be compatible with the m 
                            # Sig = Sig[:,:SKmit.shape[1]]
                            
                            # powerMask = list(powerMask)
                            # powerMask[3] = powerMask[3][:SKmit.shape[1],:]

                            #Skipping m values that are not the first m value because they are redundant for the other methods
                            
                            
                            for n_val in range(3):
                                # lower, upper = msSKThresh[m-7,n_val,:]
                                start = time.time()
                                ms_SKmit, SKarr = ms_SKmitigate(Sig_lin, n=n[n_val], d=1, m=2**m)
                                end = time.time()
                                msSK_time = end - start
                                try:
                                    msSK_TP, msSK_FP, msSK_Pr, msSK_Acc = errorCalculator(powerMask, ms_SKmit)
                                except Exception as e:
                                    print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                    continue
                                msSKds[sr,fc,fs,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP, msSK_time]

                            if m != 7:
                                continue
                            print("I am doing convRFI and AOFlagger now: m = ", m)

                            for A1i, A1 in enumerate(Agfac1):
                                for A2i, A2 in enumerate(Agfac2):
                                    for A3i, A3 in enumerate(Agfac3):
                                        for A4i, A4 in enumerate(Agfac4):
                                            for bi, b in enumerate(bins):
                                                try:
                                                    start = time.time()
                                                    ConvMit = ConvRFI_mitigate(Sig_lin, agg_factor=[A1,A2,A3,A4], bins = b)
                                                    end = time.time()
                                                    Conv_time = end - start
                                                except Exception as e:
                                                    print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                                    continue
                                                try:
                                                    Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
                                                except Exception as e:
                                                    print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                                    continue
                                                ConvRFIds[sr,fc,fs,A1i,A2i,A3i,A4i,bi,snr,dc,:] = [Conv_Pr, Conv_Acc, Conv_TP, Conv_FP, Conv_time] 
                                
                            for ci, c in enumerate(Count):
                                try:
                                    start = time.time()
                                    AoMit = aoflaggerMit(Sig_lin, count=c)
                                    end = time.time()
                                    AO_time = end - start
                                except Exception as e:
                                    print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                    continue
                                try:
                                    Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
                                except Exception as e:
                                    print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                    continue
                                AOFlaggerds[sr,fc,fs,ci,snr,dc,:] = [Ao_Pr, Ao_Acc, Ao_TP, Ao_FP, AO_time]
                            

                        # try:
                        #     # SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
                        #     # Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
                        #     # Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
                        # except Exception as e:
                        #     print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                        #     continue
                        # msSKds[sr,fc,fs,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP]
    
                print (f'Completed metrics for SymbolRate={(SymRt[sr])}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
                SKds.to_netcdf(f'home/scratch/amuthiya/QPSK_SK_{sr}_{fc}_{fs}.nc')
                msSKds.to_netcdf(f'home/scratch/amuthiya/QPSK_msSK_{sr}_{fc}_{fs}.nc')
                ConvRFIds.to_netcdf(f'home/scratch/amuthiya/QPSK_ConvRFI_{sr}_{fc}_{fs}.nc')
                AOFlaggerds.to_netcdf(f'home/scratch/amuthiya/QPSK_AOFlagger_{sr}_{fc}_{fs}.nc')
    SKds.to_netcdf('home/scratch/amuthiya/QPSK_SK_full.nc')           
    msSKds.to_netcdf('home/scratch/amuthiya/QPSK_msSK_full.nc')
    ConvRFIds.to_netcdf('home/scratch/amuthiya/QPSK_ConvRFI_full.nc')
    AOFlaggerds.to_netcdf('home/scratch/amuthiya/QPSK_AOFlagger_full.nc')
    
except Exception as e:
    print(f'Error in main loop: {e}')
    SKds.to_netcdf('home/scratch/amuthiya/QPSK_SK_partial.nc')
    msSKds.to_netcdf('home/scratch/amuthiya/QPSK_msSK_partial.nc')
    ConvRFIds.to_netcdf('home/scratch/amuthiya/QPSK_ConvRFI_partial.nc')
    AOFlaggerds.to_netcdf('home/scratch/amuthiya/QPSK_AOFlagger_partial.nc')

    
### ASK FOR SK
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
Bias2 = [0.2, 0.5, 0.75]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(FC), len(FS), len(Bias2),6, len(SNR), len(DC), 5), np.nan)
CoordsDict = {
    # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    # 'Wincut': np.arange(1, 6)*0.05,
    # 'm': m,
    'Bias2': Bias2,
    'M': np.power(2, np.arange(7, 13)),
    # 'N': np.arange(1, 6),
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
SKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

### ASK FOR msSK
emptyData = np.full((6, 4, 5, 3, 3, 6,4,6, 5), np.nan)
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
Bias2 = [0.2, 0.5, 0.75]
n = [2,4,8]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]

CoordsDict = {
    # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    # 'Wincut': np.arange(1, 6)*0.05,
    # 'm': m,
    'Bias2': Bias2,
    'n': n,
    'M': np.power(2, np.arange(7, 13)),
    # 'N': np.arange(1, 6),
    'SNR': SNR,
    'DC': DC,
    
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
msSKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

###ASK for ConvRFI
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
Bias2 = [0.2, 0.5, 0.75]
Agfac1 = [0.00, 0.45, 1.66, 3.00]
Agfac2 = [0.00, 0.45, 1.66, 3.00]
Agfac3 = [0.00, 0.45, 1.66, 3.00]
Agfac4 = [0.00, 0.45, 1.66, 3.00]
bins = [26, 80, 128, 160]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(FC), len(FS), len(Bias2), len(Agfac1), len(Agfac2), len(Agfac3), len(Agfac4), len(bins), len(SNR), len(DC), 5), np.nan)

CoordsDict = {
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    'Bias2': Bias2,
    'AggressionFactor1': Agfac1,
    'AggressionFactor2': Agfac2,
    'AggressionFactor3': Agfac3,
    'AggressionFactor4': Agfac4,
    'Bins': bins,
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
ConvRFIds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

###ASK for AOFlagger
SymRt = [1, 4,20,50,100,200]
FC = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
Bias2 = [0.2, 0.5, 0.75]
Count = [1, 5]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(FC), len(FS), len(Bias2), len(Count), len(SNR), len(DC), 5), np.nan)

CoordsDict = {
    'SymbolRate': SymRt,
    'FC': FC,
    'FS': FS,
    'Bias2': Bias2,
    'Count': Count,
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
AOFlaggerds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

### ASK
try:
    for sr in range(6):
        
        print(f'{(sr/6)} percent complete')
        
        for fc in range(4): 
            for fs in range(5):
                for b2 in range(3):  
            # if 5*sr*30*1e6 > fs*200*1e6:
            #     print(f'Skipping invalid configuration: SymbolRate={SymRt[sr]} ksps, FS={FS[fs]} MHz')
            #     continue
                    try:
                        RawVolt = VoltGen('ask', 75, SymRt[sr], ((FC[fc]*(FS[fs]/128))+120)*1e6, biases= np.array([1,Bias2[b2]]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6)
                    except Exception as e:
                        print(f'Error generating voltage signal for SymbolRate={SymRt[sr]}, fc={FC[fc]}, FS={FS[fs]}: {e}')
                        continue
                    # if 24*16*128*m > 6000*nb*fs*200*1e6/(SymRt[sr]*30*1e6):
                        #     print(f'Skipping invalid configuration: M={128*m}, nbits={6000*nb}, fs={FS[fs]} MHz, SymbolRate={SymRt[sr]} ksps')
                        #     continue
                    for snr in range(4):
                        noise = np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8) + 1.j*np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8)
                        for dc in range(6):
                            DuVolt = DuCyc(RawVolt, DC=DC[dc])
                            for m in range(7, 13):
                                print(f'Processing: SymbolRate={SymRt[sr]} ksps, fc={(FC[fc]*(FS[fs]/128)+120)*1e6} Hz, FS={FS[fs]*1e6} Hz, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
                                Sig, Sig_lin, Sig_db, powerMask = VolttoSig(DuVolt, noise, M=2**m, SNR=SNR[snr], DC=DC[dc])

                                # Sig, Sig_lin, Sig_db, powerMask = sigGen('ask', 75, SymRt[sr], ((FC[fc]*(FS[fs]/128))+120)*1e6, biases= np.array([0.5, 1]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6, M=2**m, SNR=SNR[snr], DC=DC[dc])

                                # except Exception as e:
                                #     print (f'Error converting voltage to signal for SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}: {e}')
                                #     continue

                                #Adjusting Signal length to be compatible with the m value
                                times = Sig.shape[1]
                                if times % (2**m) != 0:                                            
                                    print(f'M (time bin size) must be a factor of the number of time samples. Got M={2**m} and time samples={times}.')
                                    if 2**m > times:
                                        raise ValueError('M is larger than the number of time samples. Thats not enough samples')
                                    Sig = Sig[:,:-(times % (2**m))]
                                    print(f'Adjusting signal length to {Sig.shape[1]} for compatibility.')
                                Sig_lin = Sig_lin[:,:Sig.shape[1]]
                                Sig_db = Sig_db[:,:Sig.shape[1]]
                                powerMask = list(powerMask)
                                powerMask[3] = powerMask[3][:Sig.shape[1],:]
                                
                                start = time.time()
                                SKmit = SKmitigate(Sig_lin, n=1, d=1, m=2**m)
                                end = time.time()
                                SK_time = end - start

                                try: 
                                    SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
                                except Exception as e:
                                    print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                    continue
                                SKds[sr,fc,fs,b2,m-7,snr,dc,:] = [SK_Pr, SK_Acc, SK_TP, SK_FP, SK_time]

                                # adjust the Signal length to be compatible with the m 
                                # Sig = Sig[:,:SKmit.shape[1]]
                                
                                # powerMask = list(powerMask)
                                # powerMask[3] = powerMask[3][:SKmit.shape[1],:]

                                #Skipping m values that are not the first m value because they are redundant for the other methods
                                
                                
                                for n_val in range(3):
                                    # lower, upper = msSKThresh[m-7,n_val,:]
                                    start = time.time()
                                    ms_SKmit, SKarr = ms_SKmitigate(Sig_lin, n=n[n_val], d=1, m=2**m)
                                    end = time.time()
                                    msSK_time = end - start
                                    
                                    try:
                                        msSK_TP, msSK_FP, msSK_Pr, msSK_Acc = errorCalculator(powerMask, ms_SKmit)
                                    except Exception as e:
                                        print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                        continue
                                    msSKds[sr,fc,fs,b2,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP, msSK_time]

                                if m != 7:
                                    continue
                                print("I am doing convRFI and AOFlagger now: m = ", m)

                                for A1i, A1 in enumerate(Agfac1):
                                    for A2i, A2 in enumerate(Agfac2):
                                        for A3i, A3 in enumerate(Agfac3):
                                            for A4i, A4 in enumerate(Agfac4):
                                                for bi, b in enumerate(bins):
                                                    try:
                                                        start = time.time()
                                                        ConvMit = ConvRFI_mitigate(Sig_lin, agg_factor=[A1,A2,A3,A4], bins = b)
                                                        end = time.time()
                                                        Conv_time = end - start
                                                    except Exception as e:
                                                        print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                                        continue
                                                    try:
                                                        Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
                                                    except Exception as e:
                                                        print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                                        continue
                                                    ConvRFIds[sr,fc,fs,b2,A1i,A2i,A3i,A4i,bi,snr,dc,:] = [Conv_Pr, Conv_Acc, Conv_TP, Conv_FP, Conv_time] 
                                    
                                for ci, c in enumerate(Count):
                                    try:
                                        start = time.time()
                                        AoMit = aoflaggerMit(Sig_lin, count=c)
                                        end = time.time()
                                        AO_time = end - start
                                    except Exception as e:
                                        print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                        continue
                                    try:
                                        Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
                                    except Exception as e:
                                        print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                        continue
                                    AOFlaggerds[sr,fc,fs,b2,ci,snr,dc,:] = [Ao_Pr, Ao_Acc, Ao_TP, Ao_FP, AO_time]
                                

                            # try:
                            #     # SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
                            #     # Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
                            #     # Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
                            # except Exception as e:
                            #     print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                            #     continue
                            #     msSKds[sr,fc,fs,b2,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP, msSK_time]
        
                print (f'Completed metrics for SymbolRate={(SymRt[sr])}, fc={(FC[fc]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
                SKds.to_netcdf(f'home/scratch/amuthiya/ASK_SK_{sr}_{fc}_{fs}.nc')
                msSKds.to_netcdf(f'home/scratch/amuthiya/ASK_msSK_{sr}_{fc}_{fs}.nc')
                ConvRFIds.to_netcdf(f'home/scratch/amuthiya/ASK_ConvRFI_{sr}_{fc}_{fs}.nc')
                AOFlaggerds.to_netcdf(f'home/scratch/amuthiya/ASK_AOFlagger_{sr}_{fc}_{fs}.nc')
    SKds.to_netcdf('home/scratch/amuthiya/ASK_SK_full.nc')           
    msSKds.to_netcdf('home/scratch/amuthiya/ASK_msSK_full.nc')
    ConvRFIds.to_netcdf('home/scratch/amuthiya/ASK_ConvRFI_full.nc')
    AOFlaggerds.to_netcdf('home/scratch/amuthiya/ASK_AOFlagger_full.nc')
    
except Exception as e:
    print(f'Error in main loop: {e}')
    SKds.to_netcdf('ASK_SK_partial.nc')
    msSKds.to_netcdf('ASK_msSK_partial.nc')
    ConvRFIds.to_netcdf('ASK_ConvRFI_partial.nc')
    AOFlaggerds.to_netcdf('ASK_AOFlagger_partial.nc')


### BFSK FOR SK
SymRt = [1, 4,20,50,100,200]
F1mod = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
F2 = [125, 175, 250]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(F1mod), len(FS), len(F2),6, len(SNR), len(DC), 5), np.nan)
CoordsDict = {
    # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
    'SymbolRate': SymRt,
    'F1mod': F1mod,
    'FS': FS,
    # 'Wincut': np.arange(1, 6)*0.05,
    # 'm': m,
    'F2': F2,
    'M': np.power(2, np.arange(7, 13)),
    # 'N': np.arange(1, 6),
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
SKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

### BFSK FOR msSK
emptyData = np.full((6, 4, 5, 3, 3, 6,4,6, 5), np.nan)
SymRt = [1, 4,20,50,100,200]
F1mod = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
F2 = [125, 175, 250]
n = [2,4,8]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]

CoordsDict = {
    # 'Rmethod': ['bpsk', 'ask', 'qpsk', 'bfsk'],
    'SymbolRate': SymRt,
    'F1mod': F1mod,
    'FS': FS,
    # 'Wincut': np.arange(1, 6)*0.05,
    # 'm': m,
    'F2': F2,
    'n': n,
    'M': np.power(2, np.arange(7, 13)),
    # 'N': np.arange(1, 6),
    'SNR': SNR,
    'DC': DC,
    
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
msSKds = xr.DataArray(emptyData,  coords=CoordsDict ,dims=CoordsDict.keys())

###BFSK for ConvRFI
SymRt = [1, 4,20,50,100,200]
F1mod = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
F2 = [125, 175, 250]
Agfac1 = [0.00, 0.45, 1.66, 3.00]
Agfac2 = [0.00, 0.45, 1.66, 3.00]
Agfac3 = [0.00, 0.45, 1.66, 3.00]
Agfac4 = [0.00, 0.45, 1.66, 3.00]
bins = [26, 80, 128, 160]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(F1mod), len(FS), len(F2), len(Agfac1), len(Agfac2), len(Agfac3), len(Agfac4), len(bins), len(SNR), len(DC), 5), np.nan)

CoordsDict = {
    'SymbolRate': SymRt,
    'F1mod': F1mod,
    'FS': FS,
    'F2': F2,
    'AggressionFactor1': Agfac1,
    'AggressionFactor2': Agfac2,
    'AggressionFactor3': Agfac3,
    'AggressionFactor4': Agfac4,
    'Bins': bins,
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
ConvRFIds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

###BFSK for AOFlagger
SymRt = [1, 4,20,50,100,200]
F1mod = [0,1/8,1/4,1/2]
FS = [100,300,500,650,800]
F2 = [125, 175, 250]
Count = [1, 5]
SNR = [0.5, 1, 2, 4]
DC = [15,30,50,60,75,100]
emptyData = np.full((len(SymRt), len(F1mod), len(FS), len(F2), len(Count), len(SNR), len(DC), 5), np.nan)

CoordsDict = {
    'SymbolRate': SymRt,
    'F1mod': F1mod,
    'FS': FS,
    'F2': F2,
    'Count': Count,
    'SNR': SNR,
    'DC': DC,
    'Metrics': ['precision', 'accuracy', 'TP', 'FP', 'time']
}
AOFlaggerds = xr.DataArray(emptyData, coords=CoordsDict ,dims=CoordsDict.keys())

### BFSK
try:
    for sr in range(6):
        
        print(f'{(sr/6)} percent complete')
        
        for f1 in range(4): 
            for fs in range(5):
                for f2 in range(3):  
            # if 5*sr*30*1e6 > fs*200*1e6:
            #     print(f'Skipping invalid configuration: SymbolRate={SymRt[sr]} ksps, FS={FS[fs]} MHz')
            #     continue
                    try:
                        RawVolt = VoltGen('bfsk', 75, SymRt[sr], ((F1mod[f1]*(FS[fs]/128))+115)*1e6, biases= np.array([1,F2[f2]]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6)
                    except Exception as e:
                        print(f'Error generating voltage signal for SymbolRate={SymRt[sr]}, fc={F1mod[f1]}, FS={FS[fs]}: {e}')
                        continue
                    # if 24*16*128*m > 6000*nb*fs*200*1e6/(SymRt[sr]*30*1e6):
                        #     print(f'Skipping invalid configuration: M={128*m}, nbits={6000*nb}, fs={FS[fs]} MHz, SymbolRate={SymRt[sr]} ksps')
                        #     continue
                    for snr in range(4):
                        noise = np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8) + 1.j*np.random.RandomState().normal(0,1/SNR[snr],size=len(RawVolt)).astype(np.int8)
                        for dc in range(6):
                            DuVolt = DuCyc(RawVolt, DC=DC[dc])
                            for m in range(7, 13):
                                print(f'Processing: SymbolRate={SymRt[sr]} ksps, fc={(F1mod[f1]*(FS[fs]/128)+115)*1e6} Hz, FS={FS[fs]*1e6} Hz, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
                                Sig, Sig_lin, Sig_db, powerMask = VolttoSig(DuVolt, noise, M=2**m, SNR=SNR[snr], DC=DC[dc])

                                # Sig, Sig_lin, Sig_db, powerMask = sigGen('bfsk', 75, SymRt[sr], ((F1mod[f1]*(FS[fs]/128))+120)*1e6, biases= np.array([0.5, 1]), f1=350e6, f0=150e6, wincut=0.15, fs=FS[fs]*1e6, M=2**m, SNR=SNR[snr], DC=DC[dc])

                                # except Exception as e:
                                #     print (f'Error converting voltage to signal for SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}: {e}')
                                #     continue

                                #Adjusting Signal length to be compatible with the m value
                                times = Sig.shape[1]
                                if times % (2**m) != 0:                                            
                                    print(f'M (time bin size) must be a factor of the number of time samples. Got M={2**m} and time samples={times}.')
                                    if 2**m > times:
                                        raise ValueError('M is larger than the number of time samples. Thats not enough samples')
                                    Sig = Sig[:,:-(times % (2**m))]
                                    print(f'Adjusting signal length to {Sig.shape[1]} for compatibility.')
                                Sig_lin = Sig_lin[:,:Sig.shape[1]]
                                Sig_db = Sig_db[:,:Sig.shape[1]]
                                powerMask = list(powerMask)
                                powerMask[3] = powerMask[3][:Sig.shape[1],:]
                                
                                start = time.time()
                                SKmit = SKmitigate(Sig_lin, n=1, d=1, m=2**m)
                                end = time.time()
                                SK_time = end - start

                                try: 
                                    SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
                                except Exception as e:
                                    print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                    continue
                                SKds[sr,f1,fs,f2,m-7,snr,dc,:] = [SK_Pr, SK_Acc, SK_TP, SK_FP, SK_time]

                                # adjust the Signal length to be compatible with the m 
                                # Sig = Sig[:,:SKmit.shape[1]]
                                
                                # powerMask = list(powerMask)
                                # powerMask[3] = powerMask[3][:SKmit.shape[1],:]

                                #Skipping m values that are not the first m value because they are redundant for the other methods
                                
                                
                                for n_val in range(3):
                                    # lower, upper = msSKThresh[m-7,n_val,:]
                                    start = time.time()
                                    ms_SKmit, SKarr = ms_SKmitigate(Sig_lin, n=n[n_val], d=1, m=2**m)
                                    end = time.time()
                                    msSK_time = end - start
                                    
                                    try:
                                        msSK_TP, msSK_FP, msSK_Pr, msSK_Acc = errorCalculator(powerMask, ms_SKmit)
                                    except Exception as e:
                                        print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                        continue
                                    msSKds[sr,f1,fs,f2,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP, msSK_time]

                                if m != 7:
                                    continue
                                print("I am doing convRFI and AOFlagger now: m = ", m)

                                for A1i, A1 in enumerate(Agfac1):
                                    for A2i, A2 in enumerate(Agfac2):
                                        for A3i, A3 in enumerate(Agfac3):
                                            for A4i, A4 in enumerate(Agfac4):
                                                for bi, b in enumerate(bins):
                                                    try:
                                                        start = time.time()
                                                        ConvMit = ConvRFI_mitigate(Sig_lin, agg_factor=[A1,A2,A3,A4], bins = b)
                                                        end = time.time()
                                                        Conv_time = end - start
                                                    except Exception as e:
                                                        print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                                        continue
                                                    try:
                                                        Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
                                                    except Exception as e:
                                                        print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                                        continue
                                                    ConvRFIds[sr,f1,fs,f2,A1i,A2i,A3i,A4i,bi,snr,dc,:] = [Conv_Pr, Conv_Acc, Conv_TP, Conv_FP, Conv_time] 
                                    
                                for ci, c in enumerate(Count):
                                    try:
                                        start = time.time()
                                        AoMit = aoflaggerMit(Sig_lin, count=c)
                                        end = time.time()
                                        AO_time = end - start
                                    except Exception as e:
                                        print(f'Error mitigating signal for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                        continue
                                    try:
                                        Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
                                    except Exception as e:
                                        print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                                        continue
                                    AOFlaggerds[sr,f1,fs,f2,ci,snr,dc,:] = [Ao_Pr, Ao_Acc, Ao_TP, Ao_FP, AO_time]
                                

                            # try:
                            #     # SK_TP, SK_FP, SK_Pr, SK_Acc = errorCalculator(powerMask, SKmit)
                            #     # Conv_TP, Conv_FP, Conv_Pr, Conv_Acc = errorCalculator(powerMask, ConvMit)
                            #     # Ao_TP, Ao_FP, Ao_Pr, Ao_Acc = errorCalculator(powerMask, AoMit)
                            # except Exception as e:
                            #     print(f'Error calculating metrics for SNR={SNR[snr]}, SymbolRate={SymRt[sr]}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}: {e}')
                            #     continue
                            #     msSKds[sr,f1,fs,f2,n_val, m-7,snr,dc,:] = [msSK_Pr, msSK_Acc, msSK_TP, msSK_FP, msSK_time]
        
                print (f'Completed metrics for SymbolRate={(SymRt[sr])}, fc={(F1mod[f1]*(FS[fs]/128)+256)*1e6}, FS={FS[fs]*1e6}, M={2**m}, SNR={SNR[snr]}, DC={DC[dc]}')
                SKds.to_netcdf(f'home/scratch/amuthiya/BFSK_SK_{sr}_{f1}_{fs}.nc')
                msSKds.to_netcdf(f'home/scratch/amuthiya/BFSK_msSK_{sr}_{f1}_{fs}.nc')
                ConvRFIds.to_netcdf(f'home/scratch/amuthiya/BFSK_ConvRFI_{sr}_{f1}_{fs}.nc')
                AOFlaggerds.to_netcdf(f'home/scratch/amuthiya/BFSK_AOFlagger_{sr}_{f1}_{fs}.nc')
    SKds.to_netcdf('home/scratch/amuthiya/BFSK_SK_full.nc')           
    msSKds.to_netcdf('home/scratch/amuthiya/BFSK_msSK_full.nc')
    ConvRFIds.to_netcdf('home/scratch/amuthiya/BFSK_ConvRFI_full.nc')
    AOFlaggerds.to_netcdf('home/scratch/amuthiya/BFSK_AOFlagger_full.nc')
    
except Exception as e:
    print(f'Error in main loop: {e}')
    SKds.to_netcdf('home/scratch/amuthiya/BFSK_SK_partial.nc')
    msSKds.to_netcdf('home/scratch/amuthiya/BFSK_msSK_partial.nc')
    ConvRFIds.to_netcdf('home/scratch/amuthiya/BFSK_ConvRFI_partial.nc')
    AOFlaggerds.to_netcdf('home/scratch/amuthiya/BFSK_AOFlagger_partial.nc')