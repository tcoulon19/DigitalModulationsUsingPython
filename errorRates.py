import numpy as np
from numpy import log2,sqrt,sin,pi,exp
from scipy.special import erfc
from scipy.integrate import quad

def ser_awgn(EbN0dBs, mod_type=None, M=0, coherence=None):

    '''
    Theoretical Symbol Error Rates for various modulations over AWGN
    Parameters:
        EbN0sBs: list of SNR per bit values in dB scale
        mode_type: 'PSK', 'QAM', 'PAM', 'FSK'
        M: Modulation level for the chosen modulation
            For PSK,PAM,FSK M can be any power of 2.
            For QAM M must be even power of 2 (square QAM only)
        coherence: 'coherent' for coherent FSK detection
                   'noncoherent' for noncoherent FSK detection
            This parameter is only applicable to FSK modulation
    Returns:
        SERs: Symbol Error Rates
    '''
    if mod_type==None:
        raise ValueError('Invalid value for mod_type')
    if (M<2) or ((M & (M-1))!=0): # If M not a power of 2
        raise ValueError('M should be a power of 2')

    func_dict = {'psk': psk_awgn, 'qam': qam_awgn, 'pam': pam_awgn, 'fsk': fsk_awgn}

    gamma_s = log2(M)*(10**(EbN0dBs/10))
    if mod_type.lower() == 'fsk': # Call appropriate function
        return func_dict[mod_type.lower()](M,gamma_s,coherence)
    else:
        return func_dict[mod_type.lower{}](M,gamma_s) # Call appropriate function