def awgnPerformance():

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.special import erfc
    from modem import PSKModem, QAMModem, PAMModem, FSKModem
    from channels import awgn
    from errorRates import ser_awgn

    #--------Input Fields--------
    nSym = 10**6 # Number of symbols to transmit
    EbN0dBs = np.arange(-4,12,2) # Eb/N0 range in dB for simulation
    mod_type = 'psk' # Set 'PSK' or 'QAM' or 'FSK'
    arrayOfM = [2,4,8,16,32] # Array of M values to simulate
    #arrayOfM = [4,16,64,256] # Uncomment this line if MOD_TYPE='QAM'
    coherence = 'coherent' # 'coherent'/'noncoherent'-only for FSK

    modem_dict = {'psk': PSKModem, 'qam': QAMModem, 'pam': PAMModem, 'fsk': FSKModem}
    colors = plt.cm.jet(np.linspace(0,1,len(arrayOfM))) # Colormap
    fig, ax = plt.subplots(nrows=1,ncols=1)

    for i,M in enumerate(arrayOfM):

        #--------Initialization of various parameters--------
        k = np.log2(M)
        EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
        SER_sim = np.zeros(len(EbN0dBs)) # Simulated symbol error rates
        inputSyms = np.random.randint(low=0, high=M, size=nSym)
        # Uniform random symbols from 0 to M-1

        if mod_type.lower() == 'fsk':
            modem=modem_dict[mod_type.lower()](M,coherence) # Choose modem from dictionary
        else: # For all other modulations
            modem=modem_dict[mod_type.lower()](M) # Choose modem from dictionary
        modulatedSyms = modem.modulate(inputSyms) # Modulate

        for j,EsN0dB in enumerate(EsN0dBs):

            receivedSyms = awgn(modulatedSyms, EsN0dB) # Ad awgn noise

            if mod_type.lower() == 'fsk': # Demodulate (Refer Chapter 3)
                detectedSyms = modem.demodulate(receivedSyms,coherence)
            else: # Demodulate (Refer Chapter 2)
                detectedSyms = modem.demodulate(receivedSyms)

            SER_sim[j] = np.sum(detectedSyms != inputSyms)/nSym

        SER_theory = ser_awgn(EbN0dBs, mod_type, M, coherence) # Theory SER
        ax.semilogy(EbN0dBs, SER_sim, color=colors[i], marker='o', linestyle = '', label = 'Sim'+str(M)+'-'+mod_type.upper())
        ax.semilogy(EbN0dBs, SER_theory, color=colors[i], linestyle = '-', label='Theory'+str(M)+'-'+mod_type.upper())

    ax.set_xlabel('Eb/N0(dB)'); ax.set_ylabel('SER ($P_s$)')
    ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over AWGN')
    ax.legend(); fig.savefig("ch4_images/awgnPerformance.png")



def rayleighPerformance():

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm # Colormap
    from scipy.special import erfc
    from modem import PAMModem, PSKModem, QAMModem, FSKModem
    from channels import awgn, rayleighFading
    from errorRates import ser_rayleigh

    #--------Input Fields--------
    nSym = 10**6 # Number of symbols to transmit
    EbN0dBs = np.arange(-4,12,2) # Eb/N0 range in dB for simulation
    mod_type = 'PAM' # Set 'PSK' or 'QAM' or 'PAM'
    arrayOfM = [2,4,8,16,32] # Array of M values to simulate
    #arrayOfM = [4,16,64,256] # Uncomment this line for QAM

    modem_dict = {'psk': PSKModem, 'qam': QAMModem, 'pam': PAMModem}
    colors = plt.cm.jet(np.linspace(0,1,len(arrayOfM))) # Colormap
    fig, ax = plt.subplots(1,1)

    for i,M in enumerate(arrayOfM):

        k = np.log2(M)
        EsN0dBs = 10*np.log10(k)+EbN0dBs # EsN0dB calculation
        SER_sim = np.zeros(len(EbN0dBs)) # Simulated Symbol error rates
        # Uniform random symbols from 0 to M-1
        inputSyms = np.random.randint(0,M,size=nSym)

        modem = modem_dict[mod_type.lower()](M) # Choose a modem from the dictionary
        modulatedSyms = modem.modulate(inputSyms) # Modulate

        for j,EsN0dB in enumerate(EsN0dBs):

            h_abs = rayleighFading(nSym) # Rayleigh flat fading samples
            hs = h_abs*modulatedSyms # Fading effect on modulated symbols
            receivedSyms = awgn(hs,EsN0dB) # Add awgn noise

            y = receivedSyms/h_abs # Decision vector
            detectedSyms = modem.demodulate(y) # Demodulate
            SER_sim[j] = np.sum(detectedSyms != inputSyms)/nSym

        SER_theory = ser_rayleigh(EbN0dBs,mod_type,M) # Theory SER
        ax.semilogy(EbN0dBs,SER_sim,color=colors[i], marker='o', linestyle='', label='Sim '+str(M)+'-'+mod_type.upper())
        ax.semilogy(EbN0dBs,SER_theory,color=colors[i],linestyle='-',label='Theory '+str(M)+'-'+mod_type.upper())

    ax.set_xlabel('Eb/N0(dB)'); ax.set_ylabel('SER ($P_s$)')
    ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over Rayleigh flat fading channel')
    ax.legend(); fig.savefig('Ch4_images/rayleighPerformance.png')



def ricianPerformance():

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm # Colormap for color palette
    from scipy.special import erfc
    from modem import PSKModem, QAMModem, PAMModem, FSKModem
    from channels import awgn, ricianFading
    from errorRates import ser_rician

    #--------Input Fields--------
    nSym = 10**6 # Number of symbols to transmit
    EbN0dBs = np.arange(0,22,2) # Eb/N0 range in dB for simulation
    K_dBs = [3,5,10,20] # Array of K factors for Rician fading in dB
    mod_type = 'QAM' # Set 'PSK' or 'QAM' or 'PAM'
    M = 64 # M value for the modulation to simulate

    modem_dict = {'psk': PSKModem, 'qam': QAMModem, 'pam': PAMModem}
    colors = plt.cm.jet(np.linspace(0,1,len(K_dBs))) # Colormap
    fig, ax = plt.subplots(nrows=1,ncols=1)

    for i, K_dB in enumerate(K_dBs):
        #--------Initialization of various parameters--------
        k = np.log2(M)
        EsN0dBs = 10*np.log10(k) + EbN0dBs # EsN0dB calculation
        SER_sim = np.zeros(len(EbN0dBs)) # Simulated symbol error rates
        # Uniform random symbols from 0 to M-1
        inputSyms = np.random.randint(low=0,high=M,size=nSym)

        modem = modem_dict[mod_type.lower()](M) # Choose a modem from the dictionary
        modulatedSyms = modem.modulate(inputSyms) # Modulate

        for j, EsN0dB in enumerate(EsN0dBs):
            h_abs = ricianFading(K_dB,nSym) # Rician flat fading samples
            hs = h_abs*modulatedSyms # Fading effect on modulated symbols
            receivedSyms = awgn(hs,EsN0dB) # Add awgn noise
            y = receivedSyms/h_abs # Decision vector
            detectedSyms = modem.demodulate(y) # Demodulate
            SER_sim[j] = np.sum(detectedSyms != inputSyms)/nSym

        SER_theory = ser_rician(K_dB, EbN0dBs, mod_type,M)
        ax.semilogy(EbN0dBs,SER_sim,color=colors[i],marker='o',linestyle='',label='Sim K'+str(K_dB)+' dB')
        ax.semilogy(EbN0dBs,SER_theory,color=colors[i],linestyle='-',label='Thoery K='+str(K_dB)+' dB')
    
    ax.set_xlabel('Eb/N0(dB)'); ax.set_ylabel('SER ($P_s$)'); ax.set_ylim(10**-7,1)
    ax.set_title('Probability of Symbol Error for M-'+str(mod_type)+' over Rayleigh flat fading channel')
    ax.legend()
    fig.savefig('Ch4_images/ricianPerformance.png')


ricianPerformance()







