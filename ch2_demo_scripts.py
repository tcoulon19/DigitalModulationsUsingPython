# Performance of BPSK using waveform simulation
from re import T


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
        
        plt.figure(0)
        plt.plot(t, s_bb) # Baseband wfm zoomed to first 10 bits
        plt.xlabel('t(s)')
        plt.ylabel('$s_{bb}(t)$-baseband')
        plt.xlim(0,10*L)
        plt.title('Signal after BPSK')
        plt.savefig('Ch2_images/BPSK_performance_im1.png')

        plt.figure(1)
        plt.plot(t, s) # Transmitted wfm zoomed to first 10 bits
        plt.xlabel('t(s)')
        plt.ylabel('s(t)-with carrier')
        plt.xlim(0,10*L)
        plt.title('Signal multiplied by carrier')
        plt.savefig('Ch2_images/BPSK_performance_im2.png')

        plt.figure(2)
        plt.plot(np.real(s_bb), np.imag(s_bb), 'o')
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.title('Constellation diagram')
        plt.savefig('Ch2_images/BPSK_performance_im3.png')

        for i,EbN0 in enumerate(EbN0dB):
            
            # Compute and add AWGN noise
            r = awgn(s, EbN0, L) # Refer Chapter section 4.1

            r_bb = r*np.cos(2*np.pi*Fc*t/Fs) # Recovered baseband signal
            ak_hat = bpsk_demod(r_bb, L) # Baseband correlation demodulator
            BER[i] = np.sum(ak != ak_hat)/N # Bit Error Rate Computation (!= means "not equal to")

            # Received signal waveform zoomed to first 10 bits, EbN0dB=9
            if EbN0 == 10:

                plt.figure(3)
                plt.plot(t,r)
                plt.xlabel('t(s)')
                plt.ylabel('r(t)')
                plt.xlim(0,10*L)
                plt.title('Recieved signal with noise')
                plt.savefig('Ch2_images/BPSK_performance_im4.png')

                plt.figure(4)
                plt.plot(16*np.arange(N),ak_hat)
                plt.xlabel('t(s)')
                plt.ylabel('ak_hat')
                plt.xlim(0,10*L)
                plt.title('Demodulated signal')
                plt.savefig('Ch2_images/BPSK_performance_im5.png')

        #----------Theoretical Bit/Symbol Error Rates----------
        theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/10))) # Theoretical bit error rate
        
        #----------Plots----------
        plt.figure(5)
        plt.semilogy(EbN0dB, BER, 'k*', label='Simulated') # Simulated BER
        plt.semilogy(EbN0dB, theoreticalBER, 'r-', label='Theoretical')
        plt.xlabel('$E_b/N_0$ (dB)')
        plt.ylabel('Probability of Bit Error - $P_b$')
        plt.title('Probability of Bit Error for BPSK modulation')
        plt.savefig('Ch2_images/BPSK_performance_im6.png')


