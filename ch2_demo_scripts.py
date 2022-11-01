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
                plt.title('Recieved signal with noise, EbN0=10')
                plt.savefig('Ch2_images/BPSK_performance_im4.png')

                plt.figure(4)
                plt.plot(16*np.arange(N),ak_hat)
                plt.xlabel('t(s)')
                plt.ylabel('ak_hat')
                plt.xlim(0,10*L)
                plt.title('Demodulated signal, EbN0=10')
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


# Coherent detection of DEBPSK
def DEBPSK_performance():

    import numpy as np
    import matplotlib.pyplot as plt
    from passband_modulations import bpsk_mod, bpsk_demod
    from channels import awgn
    from scipy.signal import lfilter
    from scipy.special import erfc

    N = 1000000 # Number of symbols to transmit
    EbN0dB = np.arange(-4,11,2) # Eb/N0 range in dB for simulation
    L = 16 # Oversampling factor, L = Tb/Ts (Tb = bit period, Ts = sampling period)
    # If a carrier is used, use L = Fs/Fc, where Fs >> 2*Fc
    Fc = 800 # Carrier frequency
    Fs = L*Fc # Sampling frequency
    SER = np.zeros(len(EbN0dB)) # For SER values for each EbN0

    ak = np.random.randint(2, size = N) # Uniform random symbols from 0s and 1s
    bk = lfilter([1.0], [1.0,-1.0], ak) # IIR filter for differential encoding
    bk = bk % 2 # XOR operation is equivalent to modulo-2

    [s_bb,t] = bpsk_mod(bk,L) # BPSK modulation (waveform) - baseband
    s = s_bb*np.cos(2*np.pi*Fc*t/Fs) # DEBPSK with carrier

    for i, EbN0 in enumerate(EbN0dB):

        # Compute and add AWGN noise
        r = awgn(s,EbN0,L) # Refer to Chapter section 4.1

        phaseAmbiguity = np.pi # 180 degree phase ambiguity of Costas loop
        r_bb = r*np.cos(2*np.pi*Fc*t/Fs + phaseAmbiguity) # Recovered signal
        b_hat = bpsk_demod(r_bb, L) # Baseband correlation type demodulator
        a_hat = lfilter([1.0,1.0], [1.0], b_hat) # FIR for differential decoding
        a_hat = a_hat % 2 # Binary messages, therefor modulo-2
        SER[i] = np.sum(ak != a_hat)/N # Symbol error rate computation

    #------Theoretical Bit/Symbol Error Rates------
    EbN0lins = 10**(EbN0dB/10) # Converting dB values to linear scale
    theorySER_DPSK = erfc(np.sqrt(EbN0lins))*(1-.5*erfc(np.sqrt(EbN0lins)))
    theorySER_BPSK = .5*erfc(np.sqrt(EbN0lins))

    #------Plots------
    plt.figure(0)
    plt.semilogy(EbN0dB,SER,'k*',label='Coherent DEBPSK(sim)')
    plt.semilogy(EbN0dB,theorySER_DPSK,'r-',label='Coherent DEBPSK(theory)')
    plt.semilogy(EbN0dB,theorySER_BPSK,'b-',label='Conventional BPSK')
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Probability of Bit Error - $P_b$')
    plt.title('Probability of Bit Error for DEBPSK and BPSK over AWGN')
    plt.legend()
    plt.savefig('Ch2_images/DEBPSK_performance.png')


# DBPSK non-coherent detection
def DBPSK_noncoherent():

    import numpy as np
    import matplotlib.pyplot as plt
    from passband_modulations import bpsk_mod
    from channels import awgn
    from scipy.signal import lfilter
    from scipy.special import erfc

    N = 100000 # Number of symbols to transmit
    EbN0dB = np.arange(-4,11,2) # Eb/N0 range in dB for simulation
    L = 8 # Oversampling factor L = Tb/Ts
    # If carrier is used, use L = Fs/Fc where Fs >> 2*Fc
    Fc = 800 # Carrier frequency
    Fs = L*Fc # Sampling frequency

    BER_suboptimum = np.zeros(len(EbN0dB)) # BER measures
    BER_optimum = np.zeros(len(EbN0dB))

    #--------Transmitter--------
    ak = np.random.randint(2, size=N) # Uniform random symbols from 0s and 1s
    bk = lfilter([1.0],[1.0,-1.0],ak) # IIR filter for differential encoding
    bk = bk % 2 # XOR operation is equivalent to modulo-2
    [s_bb,t] = bpsk_mod(bk,L) # BPSK modulation (waveform) - baseband
    s = s_bb*np.cos(2*np.pi*Fc*t/Fs).astype(complex) # DBPSK with carrier

    for i,EbN0 in enumerate(EbN0dB):

        # Compute and add AWGN noise
        r = awgn(s,EbN0,L) # Refer to Chapter section 4.1

        #--------Suboptimum receiver--------
        p = np.real(r)*np.cos(2*np.pi*Fc*t/Fs) # Demodulate to baseband using BPF
        w0 = np.hstack((p,np.zeros(L))) # Append L samples on one arm for equal lengths
        w1 = np.hstack((np.zeros(L),p)) # Delay the other arm by Tb (L samples)
        w = w0*w1 # Multiplier
        z = np.convolve(w, np.ones(L)) # Integrator from kTb to (k+1)Tb (L samples)
        u = z[L-1:-1-L:L] # Sampler t = kTb
        ak_hat = (u<0) # Decicion
        BER_suboptimum[i] = np.sum(ak != ak_hat)/N # BER for suboptimum receiver

        #--------Optimum receiver--------
        p = np.real(r)*np.cos(2*np.pi*Fc*t/Fs) # Multiply I arm by cos
        q = np.imag(r)*np.sin(2*np.pi*Fc*t/Fs) # Multiply Q arm by sin
        x = np.convolve(p, np.ones(L)) # Integrate I-arm by Tb duration (L samples)
        y = np.convolve(q, np.ones(L)) # Integrate Q-arm by Tb duration (L samples)
        xk = x[L-1:-1:L] # Sample every Lth sample
        yk = y[L-1:-1:L] # Sample every Lth sample
        w0 = xk[0:-2] # Non-delayed version on I-arm
        w1 = xk[1:-1] # 1-bit delay on I-arm
        z0 = yk[0:-2] # Non-delayed version on Q-arm
        z1 = yk[1:-1] # Non-delayed version on Q-arm
        u = w0*w1 + z0*z1 # Decision statistic
        ak_hat = (u<0) # Threshold detection
        BER_optimum[i] = np.sum(ak[1:-1] != ak_hat)/N # BER for optimum receiver

    #--------Theoretical Bit/Symbol Error Rates--------
    EbN0lins = 10**(EbN0dB/10) # Convert dB to linear
    theory_DBPSK_optimum = .5*np.exp(-EbN0lins)
    theory_DBPSK_suboptimum = .5*np.exp(-.76*EbN0lins)
    theory_DBPSK_coherent = erfc(np.sqrt(EbN0lins))*(1-.5*erfc(np.sqrt(EbN0lins)))
    theory_BPSK_conventional = .5*erfc(np.sqrt(EbN0lins))

    #------Plots------
    plt.figure(0)
    plt.semilogy(EbN0dB, BER_suboptimum, 'k*', label = 'DBPSK subopt (sim)')
    plt.semilogy(EbN0dB, BER_optimum, 'b*', label = 'DBPSK opt (sim)')
    plt.semilogy(EbN0dB, theory_DBPSK_suboptimum, 'm-', label = 'DBPSK subopt (theory)')
    plt.semilogy(EbN0dB, theory_DBPSK_optimum, 'r-', label = 'DBPSK opt (theory)')
    plt.semilogy(EbN0dB, theory_DBPSK_coherent, 'k-', label = 'Coherent DEBPSK')
    plt.semilogy(EbN0dB, theory_BPSK_conventional, 'b-', label = 'Coherent BPSK')
    plt.xlabel('$E_b/N_0 (dB)$')
    plt.ylabel('$Probability of Bit Error - P_b$')
    plt.title('Probability of D-BPSK over AWGN')
    plt.legend()
    plt.savefig('Ch2_images/DBPSK_noncoherent.png')

DBPSK_noncoherent()







