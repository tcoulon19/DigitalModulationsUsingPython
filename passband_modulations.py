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



