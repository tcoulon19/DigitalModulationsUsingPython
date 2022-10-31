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

        x = np.real(r_bb) # I arm
        x = np.convolve(x, np.ones(L)) # integrate for Tb duration (L samples)
        x = x[L-1:-1:L] # I arm - sample at every L
        ak_hat = (x > 0).transpose() # threshold detector
        
        return ak_hat 


