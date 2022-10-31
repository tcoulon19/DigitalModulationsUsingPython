# Performance of BPSK using waveform simulation
def BPSK_performance():

        import numpy as np
        import matplotlib.pyplot as plt
        from passband_modulations import bpsk_mod, bpsk_demod
        from channels import awgn
        from scipy.special import erfc

        N=10000 # Number of symbols to transmit
        EbN0dB = np.arange(-4,11,2) # Eb/N0 range in dB for simulation
        L=16 # Oversampling factor, L = Tb/Ts (Tb = bit period, Ts = sampling period)
        # If carrier is used, use L = Fs/Fc, where Fs >> 2xFc
        Fc = 800 # Carrier frequency
        Fs = L*Fc # Sampling frequency
        BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0
        ak = np.random.randint(2, size=N) # Uniform random symbols from 0s and 1s
        (s_bb, t) = bpsk_mod(ak,L) # BPSK modulation (waveform) - baseband
        s = s_bb*np.cos(2*np.pi*Fc*t/Fs) # With carrier
        # Waveforms at the transmitter
        