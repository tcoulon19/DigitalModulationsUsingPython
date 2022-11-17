# BPSK baseband only, can be multiplied by carrier freq outside function for passband
from re import S
from matplotlib.artist import ArtistInspector
import numpy as np


def bpsk_mod(ak, L):

    import numpy as np
    from scipy.signal import upfirdn

    '''
    Function to modulate an incoming bindry stream using BPSK (baseband)
    Parameters:
        ak: input binary data stream (0s and 1s) to modulate
        L: oversampling factor (Tb/Ts)
    Returns:
        (s_bb,t): tuple of following variables
            s_bb: BPSK modulated signal (baseband) -- s_bb(t)
            t: generated time base for modulated signal
    '''

    s_bb = upfirdn(h=[1]*L, x=2*ak-1, up=L) # NRZ encoder (+1V for 1 and -1V for 0)
    t=np.arange(0, len(ak)*L) # discrete time base

    return (s_bb, t)


# Baseband BPSK detection
def bpsk_demod(r_bb, L):

    import numpy as np

    '''
    Function to demodulate a BPSK (baseband) signal
    Parameters:
        r_bb: received signal at the receiver front end (baseband)
        L: oversampling factor (Tsym/Ts)
    Returns:
        ak_hat: detected/estimated binary stream
    '''

    x = np.real(r_bb) # I arm
    x = np.convolve(x, np.ones(L)) # integrate for Tb duration (L samples)
    x = x[L-1:-1:L] # I arm - sample at every L
    ak_hat = (x > 0).transpose() # threshold detector
    
    return ak_hat 


# QPSK modulator. Note: Plots of I(t) and Q(t) plot I and Q, not I_t, and Q_t
def qpsk_mod(a, fc, OF, enable_plot = False):

    import numpy as np
    import matplotlib.pyplot as plt

    '''
    Modulate an incoming binary stream using conventional QPSK
    Parameters:
        a: input binary data stream (0s and 1s) to modulate
        fc: carrier frequency in Hz
        OF: oversampling factor -- at least 4 is better
        enable_plot: True = plot transmitter waveforms (default False)
    Returns:
        result: Dictionary containing the following keyword entries:
        s(t): QPSK modulated signal vector with carrier i.e. s(t)
        I(t): baseband I channel waveform (no waveform)
        Q(t): baseband Q channel waveform (no carrier)
        t: time base for the carrier modulated signal
    '''
    L = 2*OF # Samples in each symbol (QPSK has 2 bits per symbol)
    I = a[0::2]; Q = a[1::2] # Evem amd odd bit streams
    # Even/odd streams at 1/2Tb baud. Note: Baud rate is rate at which infromation is transferred in a communication channel

    from scipy.signal import upfirdn # NRZ encoder
    I = upfirdn(h=[1]*L, x=2*I-1, up=L)
    Q = upfirdn(h=[1]*L, x=2*Q-1, up=L)
    fs = OF*fc # Sampling frequency
    t = np.arange(0,len(I)/fs,1/fs) # Time base

    I_t = I*np.cos(2*np.pi*fc*t); Q_t = -Q*np.sin(2*np.pi*fc*t)
    s_t = I_t + Q_t # QPSK modulated signal

    if enable_plot:

        plt.figure(0)
        plt.clf()
        plt.plot(t,I)
        plt.xlim(0,20*L/fs)
        plt.title('I(t)')
        plt.savefig('Ch2_images/qpsk_mod_im1')

        plt.figure(1)
        plt.clf()
        plt.plot(t,Q)
        plt.xlim(0,20*L/fs)
        plt.title('Q(t)')
        plt.savefig('Ch2_images/qpsk_mod_im2')

        plt.figure(2)
        plt.clf()
        plt.plot(t,I_t,'r')
        plt.xlim(0,20*L/fs)
        plt.title('$I(t) cos(2 \pi f_c t)$')
        plt.savefig('Ch2_images/qpsk_mod_im3')

        plt.figure(3)
        plt.clf()
        plt.plot(t,Q_t,'r')
        plt.xlim(0,20*L/fs)
        plt.title('$Q(t) sin(2 \pi f_c t)$')
        plt.savefig('Ch2_images/qpsk_mod_im4')

        plt.figure(4)
        plt.clf()
        plt.plot(t,s_t)
        plt.xlim(0,20*L/fs)
        plt.title('$s(t) = I(t) cos(2 \pi f_c t) - Q(t) sin(2 \pi f_c t)$')
        plt.savefig('Ch2_images/qpsk_mod_im5')

    result = dict()
    result['s(t)'] = s_t
    result['I(t)'] = I
    result['Q(t)'] = Q
    result['t'] = t

    return result


# QPSK demodulator
def qpsk_demod(r, fc, OF, enable_plot=False):

    import numpy as np
    import matplotlib.pyplot as plt

    '''
    Demodulate a conventional QPSK signal
    Parameters:
        r: received signal at the receiver front end
        fc: carrier frequency (Hz)
        OF: oversampling factor (at least 4 is better)
        enable_plot: True = plot receiver waveforms (default False)
    Returns:
        a_hat: detected binary stream
    '''
    fs = OF*fc # Sampling frequency
    L = 2*OF # Number of samples in 2Tb duration
    t = np.arange(0,len(r)/fs,1/fs) # Time base
    x = r*np.cos(2*np.pi*fc*t) # I arm
    y = -r*np.sin(2*np.pi*fc*t) # Q arm
    x = np.convolve(x,np.ones(L)) # Integrate for L (Tsym=2*Tb) duration
    y = np.convolve(y,np.ones(L)) # Integrate for L (Tsym=2*Tb) duration

    x = x[L-1::L] # I arm -- sample at every symbol instant Tsym
    y = y[L-1::L] # Q arm -- sample at every symbol instant Tsym
    a_hat = np.zeros(2*len(x))
    a_hat[0::2] = (x>0) # Even bits
    a_hat[1::2] = (y>0) # Odd bits

    if enable_plot:

        plt.figure(5)
        plt.clf()
        plt.plot(x[0:200],y[0:200],'o')
        plt.title('Demodulated signal constelation plot')
        plt.savefig('Ch2_images/qpsk_demod')

    return a_hat


# OQPSK modulator. Note: Plots of I(t) and Q(t) plot I and Q, not I_t, and Q_t
def oqpsk_mod(a, fc, OF, enable_plot=False):

    import numpy as np
    import matplotlib.pyplot as plt

    '''
    Modulate an incoming binary stream using OQPSK
    Parameters:
        a: input binary stream (0s and 1s) to modulate
        fc: carrier freq in Hz
        OF: oversampling factor -- at least 4 is better
        enable_plot: True = plot transmitter waveforms (default False)
    Returns:
        result: dictionary containing the following keyword entries:
            s(t): QPSK modulated signal vector with carrier
            I(t): baseband I channel waveform (no carrier)
            Q(t): baseband Q channel waveform (no carrier)
            t: time base for the carrier modulated signal
    '''
    L = 2*OF # Samples in each symbol (QPSK has 2 bits per symbol)
    I = a[0::2]; Q=a[1::2] # Even and odd bit streams
    # Even/odd streams at 1/2Tb baud
    from scipy.signal import upfirdn # NRZ encoder
    I = upfirdn(h=[1]*L, x=2*I-1, up=L)
    Q = upfirdn(h=[1]*L, x=2*Q-1, up=L)

    I = np.hstack((I,np.zeros(L//2))) # 0-padding at end
    Q = np.hstack((np.zeros(L//2),Q)) # 0-padding at start

    fs = OF*fc # Sampling frequency
    t = np.arange(0,len(I)/fs,1/fs) # Time base
    I_t = I*np.cos(2*np.pi*fc*t); Q_t = -Q*np.sin(2*np.pi*fc*t)
    s_t = I_t + Q_t # QPSK modulated baseband signal

    if enable_plot:

        plt.figure(0)
        plt.clf()
        plt.plot(t,I)
        plt.xlim(0,20*L/fs)
        plt.title('I(t)')
        plt.savefig('Ch2_images/oqpsk_mod_im1')

        plt.figure(1)
        plt.clf()
        plt.plot(t,Q)
        plt.xlim(0,20*L/fs)
        plt.title('Q(t)')
        plt.savefig('Ch2_images/oqpsk_mod_im2')

        plt.figure(2)
        plt.clf()
        plt.plot(t,I_t,'r')
        plt.xlim(0,20*L/fs)
        plt.title('$I(t) cos(2 \pi f_c t)$')
        plt.savefig('Ch2_images/oqpsk_mod_im3')

        plt.figure(3)
        plt.clf()
        plt.plot(t,Q_t,'r')
        plt.xlim(0,20*L/fs)
        plt.title('$Q(t) sin(2 \pi f_c t)$')
        plt.savefig('Ch2_images/oqpsk_mod_im4')

        plt.figure(4)
        plt.clf()
        plt.plot(t,s_t)
        plt.xlim(0,20*L/fs)
        plt.title('$s(t) = I(t) cos(2 \pi f_c t) - Q(t) sin(2 \pi f_c t)$')
        plt.savefig('Ch2_images/oqpsk_mod_im5')

    result = dict()
    result['s(t)'] = s_t
    result['I(t)'] = I
    result['Q(t)'] = Q
    result['t'] = t

    return result


# OQPSK demodulator
def oqpsk_demod(r, N, fc, OF, enable_plot=False):

    import numpy as np
    import matplotlib.pyplot as plt

    '''
    Demodulate a OQPSK signal
    Parameters:
        r: received signal at the receiver front end
        N: Number of OQPSK symbols transmitted
        fc: carrier frequency (Hz)
        OF: oversampling factor (at least 4 is better)
        enable_plot: True = plot receiver waveforms (default False)
    Returns:
        a_hat: detected binary stream
    '''
    fs = OF*fc # Sampling frequency
    L = 2*OF # Number of samples in 2Tb duration
    t = np.arange(0,(N+1)*OF/fs,1/fs) # Time base
    x = r*np.cos(2*np.pi*fc*t) # I arm
    y = -r*np.sin(2*np.pi*fc*t) # Q arm
    x = np.convolve(x,np.ones(L)) # Integrate for L (Tsym=2*Tb) duration
    y = np.convolve(y,np.ones(L)) # Integrate for L (Tsym=2*Tb) duration
    
    x = x[L-1:-L+L//2:L] # I arm - sample at every symbol instant Tsym
    y = y[L+L//2-1:-1-L//2:L] # Q arm - sample at every symbol starting at L+L/2-1th sample

    a_hat = np.zeros(N)
    a_hat[0::2] = (x>0) # Even bits
    a_hat[1::2] = (y>0) # Odd bits

    if enable_plot:

        plt.figure(5)
        plt.clf()
        plt.plot(x[0:200],y[0:200],'o')
        plt.title('Demodulated signal constelation plot')
        plt.savefig('Ch2_images/oqpsk_demod')
    
    return a_hat


# Differential encoding for pi/4-DQPSK
def piBy4_dqpsk_diff_encoding(a, enable_plot=False):

    import numpy as np
    import matplotlib.pyplot as plt

    '''
    Phase mapper for pi/4-DQPSK modulation
    Parameters:
        a: input stream of binary bits
    Returns:
        (u,v): tuple, where:
            u: differentially encoded I-channel bits
            v: differentially encoded Q-channel bits
    '''
    if len(a)%2: raise ValueError('Length of binary stream must be even')
    I = a[0::2] # Even bit stream
    Q = a[1::2] # Odd bit stream
    # Club 2-bits to form a symbol and use it as index for dTheta table
    m = 2*I+Q
    dTheta = np.array([-3*np.pi/4, 3*np.pi/4, -np.pi/4, np.pi/4]) # Lookup table for pi/4-DQPSK
    u = np.zeros(len(m)+1)
    v = np.zeros(len(m)+1)
    u[0]=1; v[0]=0 # Initial conditions for uk and vk

    for k in range(0,len(m)):

        u[k+1] = u[k] * np.cos(dTheta[m[k]]) - v[k] * np.sin(dTheta[m[k]])
        v[k+1] = u[k] * np.sin(dTheta[m[k]]) + v[k] * np.cos(dTheta[m[k]])
    
    if enable_plot: # Constellation plot

        plt.figure(0)
        plt.clf()
        plt.plot(u,v,'o')
        plt.title('Constellation')
        plt.savefig('Ch2_images/piBy4_dqpsk_diff_encoding.png')

    return (u,v)


# Pi/4-DQPSK modulator
def piBy4_dqpsk_mod(a,fc,OF,enable_plot=False):

    import numpy as np
    import matplotlib.pyplot as plt

    '''
    Modulate a binary stream using pi/4-DQPSK
    Parameters:
        a: input binary data stream (0s and 1s) to modulate
        fc: carrier frequency in Hz
        OF: oversampling factor
    Returns:
        result: Dictionary containing the following keyword entries:
            s(t): pi/4-DQPSK modulated signal vector with carrier
            U(t): differentially coded I channel waveform (no carrier)
            V(t): differentially coded Q-channel waveform (no carrier)
            t: time base
    '''
    (u,v) = piBy4_dqpsk_diff_encoding(a,enable_plot=True) # Differential encoding for pi/4-DQPSK
    # Waveform formation (similar to conventional QPSK)
    L = 2*OF # Number of samples in each symbol (QPSK has 2 bits/symbol)
    U = np.tile(u, (L,1)).flatten('F') # Odd bit stream at 1/2Tb baud
    V = np.tile(v, (L,1)).flatten('F') # Even bit steam at 1/2Tb baud

    fs = OF*fc # Sampling frequency
    t=np.arange(0, len(U)/fs,1/fs) # Time base
    U_t = U*np.cos(2*np.pi*fc*t)
    V_t = -V*np.sin(2*np.pi*fc*t)
    s_t = U_t + V_t

    if enable_plot:

        plt.figure(1)
        plt.clf()
        plt.plot(t,U)
        plt.xlim(0,10*L/fs)
        plt.title('U(t)-baseband')
        plt.savefig('Ch2_images/piBy4_dqpsk_mod_im1')

        plt.figure(2)
        plt.clf()
        plt.plot(t,V)
        plt.xlim(0,10*L/fs)
        plt.title('V(t)-baseband')
        plt.savefig('Ch2_images/piBy4_dqpsk_mod_im2')

        plt.figure(3)
        plt.clf()
        plt.plot(t,U_t,'r')
        plt.xlim(0,10*L/fs)
        plt.title('U(t)-with carrier')
        plt.savefig('Ch2_images/piBy4_dqpsk_mod_im3')

        plt.figure(4)
        plt.clf()
        plt.plot(t,V_t,'r')
        plt.xlim(0,10*L/fs)
        plt.title('V(t)-with carrier')
        plt.savefig('Ch2_images/piBy4_dqpsk_mod_im4')

        plt.figure(5)
        plt.clf()
        plt.plot(t,s_t)
        plt.xlim(0,10*L/fs)
        plt.title('s(t)')
        plt.savefig('Ch2_images/piBy4_dqpsk_mod_im5')

    result = dict()
    result['s(t)'] = s_t 
    result['U(t)'] = U
    result['V(t)'] = V
    result['t'] = t

    return result


# DQPSK differential decoding detection
def piBy4_dqpsk_diff_decoding(w,z):

    '''
    Phase mapper for pi/4-DQPSK modulation
    Parameters:
        w - differentially coded I-channel bits at the receiver
        z - differentially coded Q-channel bits at the receiver
    Returns:
        a_hat - binary bit stream after differential decoding
    '''

    if len(w) != len(z): raise ValueError('Length mismatch between w and z')

    x = np.zeros(len(w)-1)
    y = np.zeros(len(w)-1)

    for k in range(0,len(w)-1):
        x[k] = w[k+1]*w[k] + z[k+1]*z[k]
        y[k] = z[k+1]*w[k] - w[k+1]*z[k]

    a_hat = np.zeros(2*len(x))
    a_hat[0::2] = (x>0) # Odd bits
    a_hat[1::2] = (y>0) # Even bits

    return a_hat


# pi/4 -- DQPSK demodulator
def piBy4_dqpsk_demod(r,fc,OF,enable_plot=False):
    
    import matplotlib.pyplot as plt

    '''
    Differential coherent demodulation of pi/4-DQPSK
    Parameters:
        r: received signal at the receiver front end
        fc: carrier frequency in Hz
        OF: oversampling factor (multiples of fc) - at least 4 is better
    Returns:
        a_cap: detected binary stream
    '''

    fs = OF*fc # Sampling frequency
    L = 2*OF # Samples in 2Tb duration
    t = np.arange(0,len(r)/fs,1/fs)
    w = r*np.cos(2*np.pi*fc*t) # I arm
    z = -r*np.sin(2*np.pi*fc*t) # Q arm
    w = np.convolve(w,np.ones(L)) # Integrate for L (Tsym = 2*Tb) duration
    z = np.convolve(z,np.ones(L)) # Integrate for L (Tsym = 2*Tb) duration
    w = w[L-1::L] # I arm - sample at every symbol instant Tsym
    z = z[L-1::L] # Q arm - sample at every symbol instant Tsym
    a_cap = piBy4_dqpsk_diff_decoding(w,z)

    if enable_plot: # Constellation plot

        plt.figure(6)
        plt.clf()
        plt.plot(w,z,'o')
        plt.title('Constellation')
        plt.savefig('Ch2_images/piBy4_dqpsk_demod')

    return a_cap


# MSK Modulator
def msk_mod(a,fc,OF,enable_plot=False):

    import matplotlib.pyplot as plt

    '''
    Modulate an incoming binary stream using MSK
    Parameters:
        a: input binary data stream (0s and 1s) to modulate
        fc: carrier frequency in Hz
        OF: oversampling factor (at least 4 is better)
    Returns:
        result: Dictionary containing the following keyword entries:
            s(t): MSK modulated signal with carrier
            sI(t): baseband I channel waveform (no carrier)
            sQ(t): baseband Q channel waveform (no carrier)
            t: time base
    '''

    ak = 2*a-1 # NRZ encoding; 0 -> -1, 1 -> +1
    ai = ak[0::2]; aq = ak[1::2] # Split even and odd bit streams
    L = 2*OF # Represents one symbol duration Tsym=2xTb

    # Upsample by L the bits streams in I and Q arms
    from scipy.signal import upfirdn, lfilter
    ai = upfirdn(h=[1], x=ai, up=L)
    aq = upfirdn(h=[1], x=aq, up=L)

    aq = np.pad(aq, (L//2,0), 'constant') # Delay aq by Tb (delay by L/2)
    ai = np.pad(ai, (0,L//2), 'constant') # Padding at end to equal length of Q

    # Construct low-pass filter and filter the I/Q samples through it
    Fs = OF*fc
    Ts = 1/Fs
    Tb = OF*Ts
    t = np.arange(0,2*Tb+Ts,Ts)
    h = np.sin(np.pi*t/(2*Tb)) # LPF filter
    sI_t = lfilter(b=h, a=[1], x=ai) # Baseband I-channel
    sQ_t = lfilter(b=h, a=[1], x=aq) # Baseband Q-channel

    t = np.arange(0, Ts*len(sI_t), Ts) # For RF carrier
    sIc_t = sI_t*np.cos(2*np.pi*fc*t) # With carrier
    sQc_t = -sQ_t*np.sin(2*np.pi*fc*t) # With carrier
    s_t = sIc_t + sQc_t # Bandpass MSK modulated signal

    if enable_plot:

        plt.figure(0)
        plt.clf()
        plt.plot(t, sI_t, '--')
        plt.plot(t,sIc_t,'r')
        plt.xlim(-Tb,20*Tb)
        plt.title('$s_I(t)$')
        plt.savefig('Ch2_images/msk_mod_im1')

        plt.figure(1)
        plt.clf()
        plt.plot(t, sQ_t, '--')
        plt.plot(t,sQc_t,'r')
        plt.xlim(-Tb,20*Tb)
        plt.title('$s_Q(t)$')
        plt.savefig('Ch2_images/msk_mod_im2')

        plt.figure(2)
        plt.clf()
        plt.plot(t,s_t,'--')
        plt.xlim(-Tb,20*Tb)
        plt.title('s(t)')
        plt.savefig('Ch2_images/msk_mod_im3')

    result = dict()
    result['s(t)']=s_t;result['sI(t)']=sI_t;result['sQ(t)']=sQ_t;result['t']=t
    
    return result


# MSK demodulator
def msk_demod(r,N,fc,OF):

    '''
    MSK demodulator
    Parameters:
        r: received signal at the receiver front end
        N: number of symbols transmitted
        fc: carrier frequency in Hz
        OF: oversampling factor (at least 4 is better)
    Returns:
        a_hat: detected binary stream
    '''

    L = 2*OF # Samples in 2Tb duration
    Fs=OF*fc; Ts=1/Fs; Tb=OF*Ts # Sampling frequency, durations
    t = np.arange(-OF, len(r)-OF)/Fs # Time base

    # Cosine and sine functions for hald-sinusoid shaping
    x = abs(np.cos(np.pi*t/(2*Tb)))
    y = abs(np.sin(np.pi*t/(2*Tb)))

    u = r*x*np.cos(2*np.pi*fc*t) # Multiply I by half cosines and cos(2*pi*fc*t)
    v = -r*y*np.sin(2*np.pi*fc*t) # Multiply Q by half sines and sin(2*pi*fc*t)

    iHat = np.convolve(u,np.ones(L)) # Integrate for L (Tsym=2*Tb) duration
    qHat = np.convolve(v,np.ones(L)) # Integrate for L (Tsym=2*Tb) duration

    iHat = iHat[L-1::L] # I-sample at the end of every symbol
    qHat = qHat[L+L//2-1::L] # Q-sample from L+L/2th sample

    a_hat = np.zeros(N)
    a_hat[0::2] = iHat > 0 # Thresholding - odd bits
    a_hat[1::2] = qHat > 0 # Thresholding - even bits

    return a_hat


# Implementation of GMSK modulator
def gmsk_mod(a,fc,L,BT,enable_plot=False):

    '''
    Function to modulate a binary stream using GMSK
        BT: BT product (bandwidth*bit period) for GMSK
        a: input binary data stream (0s and 1s) to modulate
        fc: RF carrier frequency in Hz
        L: oversampling factor
        enable_plot: True = plot transmitter waveforms (default False)
    Returns:
        (s_t,s_complex): tuple containing the following variables
            s_t: GMSK modulated signal with carrier s(t)
            s_complex: baseband GMSK signal (I+jQ)
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import upfirdn, lfilter
    from pulseshapers import gaussianLPF

    fs = L*fc; Ts=1/fs; Tb = L*Ts; # Derived waveform timing parameters
    c_t = upfirdn(h=[1]*L, x=2*a-1, up=L) # NRZ pulse train c(t)

    k=1 # Truncation length for Gaussian LPF
    h_t = gaussianLPF(BT,Tb,L,k) # Gaussian LPF with BT=.25
    b_t = np.convolve(h_t,c_t,'full') # Convolve c(t) with Gaussian LPF to get b(t)
    bnorm_t = b_t/max(abs(b_t)) # Normalize the output of Gaussian LPF to +/-1

    h = .5 # Modulation index (peak-to-peak frequency deviation / bit rate)
    # Integrate to get phase information
    phi_t = lfilter(b=[1],a=[1,-1], x=bnorm_t*Ts) * h*np.pi/Tb

    I = np.cos(phi_t)
    Q = np.sin(phi_t) # Cross-correlated baseband I/Q signals
    s_complex = I - 1j*Q # Complex baseband representation
    t = Ts*np.arange(0,len(I)) # Time base for RF carrier
    sI_t = I*np.cos(2*np.pi*fc*t)
    sQ_t = Q*np.sin(2*np.pi*fc*t)
    s_t = sI_t + sQ_t # s(t) -- GMSK with RF carrier

    if enable_plot:

        plt.figure(0)
        plt.clf()
        plt.plot(np.arange(0,len(c_t))*Ts,c_t)
        plt.xlim(0,40*Tb)
        plt.title('c(t)')
        plt.savefig('Ch2_images/gmsk_mod_im1')

        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(-k*Tb,k*Tb+Ts,Ts),h_t)
        plt.title('$h(t): BT_b$='+str(BT))
        plt.savefig('Ch2_images/gmsk_mod_im2')

        plt.figure(2)
        plt.clf()
        plt.plot(np.arange(0,len(bnorm_t))*Ts,bnorm_t)
        plt.xlim(0,40*Tb)
        plt.title('b(t)')
        plt.savefig('Ch2_images/gmsk_mod_im3')

        plt.figure(3)
        plt.clf()
        plt.plot(np.arange(0,len(phi_t))*Ts, phi_t)
        plt.title('$phi(t)$')
        plt.savefig('Ch2_images/gmsk_mod_im4')
        
        plt.figure(4)
        plt.clf()
        plt.plot(t,I,'--')
        plt.plot(t,sI_t,'r')
        plt.xlim(0,10*Tb)
        plt.title('$I(t)cos(2 pi f_c t)$')
        plt.savefig('Ch2_images/gmsk_mod_im5')

        plt.figure(5)
        plt.clf()
        plt.plot(t,Q,'--')
        plt.plot(t,sQ_t,'r')
        plt.xlim(0,10*Tb)
        plt.title('$Q(t)sin(2 pi f_c t)$')
        plt.savefig('Ch2_images/gmsk_mod_im6')

        plt.figure(6)
        plt.clf()
        plt.plot(t,s_t)
        plt.xlim(0,20*Tb)
        plt.title('s(t)')
        plt.savefig('Ch2_images/gmsk_mod_im7')

        plt.figure(7)
        plt.clf()
        plt.plot(I,Q)
        plt.title('Constellation')
        plt.savefig('Ch2_images/gmsk_mod_im8')

    return (s_t, s_complex)


# Implementation of GMSK demodulator (assumed input is baseband already)
def gmsk_demod(r_complex,L):

    '''
    Function to demodulate a baseband GMSK signal
    Parameters:
        r_complex: received signal at receiver front end (complex form -- I+jQ)
        L: oversampling factor
    Returns:
        a_hat: detected binary stream
    '''
    I = np.real(r_complex); Q = -np.imag(r_complex); # I,Q streams
    z1 = Q*np.hstack((np.zeros(L), I[0:len(I)-L]))
    z2 = I*np.hstack((np.zeros(L), Q[0:len(I)-L]))
    z = z1 - z2
    a_hat = (z[2*L-1:-L:L]>0).astype(int) # Sampling and hard decision
    # Sampling indices depend on the truncation length (k) of the Gaussian LPF defined in the modulator

    return a_hat


# Coherent & non-coherent discrete-time BFSK
def bfsk_mod(a,fc,fd,L,fs,fsk_type='coherent', enable_plot=False):

    '''
    Function to modulate an incoming binary stream using BFSK
    Parameters:
        a: input binary data strea (0s and 1s) to modulate
        fc: center frequency of the carrier in Hz
        fd: frequency separation measured from Fc
        L: number of samples in 1-bit period
        fs: sampling frequency for discrete-time simulation
        fsk_type: 'coherent' (default) or 'non-coherent' FSK generation
        enable_plot: True = plot transmitter waveforms (default False)
    Returns:
        (s_t,phase): tuple containing following parameters
            s_t: BFSK modulated signal
            phase: initial phase generated by modulator, applicable only for coherent FSK.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import upfirdn

    a_t = upfirdn(h=[1]*L,x=a,up=L) # Data to waveform
    t = np.arange(0,len(a_t))/fs # Time base

    if fsk_type == 'noncoherent':
        # Carrier 1 with random phase
        c1 = np.cos(2*np.pi*(fc+fd/2)*t+2*np.pi*np.random.random_sample())
        # Carrier 2 with random phase
        c2 = np.cos(2*np.pi*(fc-fd/2)*t+2*np.pi*np.random.random_sample())
    else:
        # Random phase from uniform distribution (0,2pi)
        phase = 2*np.pi*np.random.random_sample()
        c1 = np.cos(2*np.pi*(fc+fd/2)*t+phase) # Carrier 1 with random phase
        c2 = np.cos(2*np.pi*(fc-fd/2)*t+phase) # Carrier 2 with the same random phase
    
    s_t = a_t*c1+(-a_t+1)*c2 # BFSK signal (MUX selection)

    if enable_plot:

        plt.figure(0)
        plt.clf()
        plt.plot(t,a_t)
        plt.xlim(0,.1)
        plt.savefig('Ch2_images/bfsk_mod_im1')

        plt.figure(1)
        plt.clf()
        plt.plot(t,s_t)
        plt.xlim(0,.01)
        plt.savefig('Ch2_images/bfsk_mod_im2')

    return (s_t,phase)


# Coherent demodulator for coherent BFSK
def bfsk_coherent_demod(r_t,phase,fc,fd,L,fs):

    '''
    Coherent demodulation of BFSK modulated signal
    Parameters:
        r_t: BFSK modulated signal at the receiver r(t)
        phase: initial phase generated at the transmitter
        fc: center frequency of the carrier in Hz
        fd: frequency separation measured from Fc
        L: number of samples in 1-bit period
        fs: sampling frequency for discrete-time simulation
    Returns:
        a_hat: data bits after demodulation
    '''

    t = np.arange(0,len(r_t))/fs # Time base
    x = r_t*(np.cos(2*np.pi*(fc+fd/2)*t+phase)-np.cos(2*np.pi*(fc-fd/2)*t+phase))
    y = np.convolve(x,np.ones(L)) # Integrate/sum from 0 to L
    a_hat = (y[L-1::L]>0).astype(int) # Sample at every sampling instant and detect

    return a_hat


# Square-law based non-coherent demodulator
def bfsk_noncoherent_demod(r_t,fc,fd,L,fs):

    '''
    Non-coherent demodulation of BFSK modulated signal
    Parameters:
        r_t: BFSK modulated signal at the receiver r(t)
        fc: center frequency of the carrier in Hz
        fd: frequency separation measured from Fc
        L: number of samples in 1-bit period
        fs: sampling frequency for discrete-time simulation
    Returns:
        a_hat: data bits after demodulation
    '''
    t = np.arange(0,len(r_t))/fs # Time base
    f1 = (fc+fd/2) 
    f2 = (fc-fd/2)

    # Define four basis functions
    p1c = np.cos(2*np.pi*f1*t)
    p2c = np.cos(2*np.pi*f2*t)
    p1s = -1*np.sin(2*np.pi*f1*t)
    p2s = -1*np.sin(2*np.pi*f2*t)

    # Multiply and integrate from 0 to L
    r1c = np.convolve(r_t*p1c,np.ones(L))
    r2c = np.convolve(r_t*p2c,np.ones(L))
    r1s = np.convolve(r_t*p1s,np.ones(L))
    r2s = np.convolve(r_t*p2s,np.ones(L))

    # Sample at every sampling instant
    r1c = r1c[L-1::L]
    r2c = r2c[L-1::L]
    r1s = r1s[L-1::L]
    r2s = r2s[L-1::L]

    # Square and add
    x = r1c**2 + r1s**2
    y = r2c**2 + r2s**2
    a_hat = ((x-y)>0).astype(int) # Compare and decide

    return a_hat

    









