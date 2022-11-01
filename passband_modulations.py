# BPSK baseband only, can be multiplied by carrier freq outside function for passband
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


# QPSK modulator
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
        plt.plot(t,I)
        plt.xlim(0,20*L/fs)
        plt.title('I(t)')
        plt.savefig('Ch2_images/qpsk_mod_im1')

        plt.figure(1)
        plt.plot(t,Q)
        plt.xlim(0,20*L/fs)
        plt.title('Q(t)')
        plt.savefig('Ch2_images/qpsk_mod_im2')

        plt.figure(2)
        plt.plot(t,I_t,'r')
        plt.xlim(0,20*L/fs)
        plt.title('$I(t) cos(2 \pi f_c t)$')
        plt.savefig('Ch2_images/qpsk_mod_im3')

        plt.figure(3)
        plt.plot(t,Q_t,'r')
        plt.xlim(0,20*L/fs)
        plt.title('$Q(t) sin(2 \pi f_c t)$')
        plt.savefig('Ch2_images/qpsk_mod_im4')

        plt.figure(4)
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







