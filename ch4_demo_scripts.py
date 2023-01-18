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
    mod_type = 'FSK' # Set 'PSK' or 'QAM' or 'FSK'
    arrayOfM = [2,4,8,16,32] # Array of M values to simulate
    #arrayOfM = [4,16,64,256] # Uncomment this line if MOD_TYPE='QAM'
    coherence = 'coherent' # 'coherent'/'noncoherent'-only for FSK

    modem_dict = {'psk': PSKModem, 'qam': QAMModem, 'pam': PAMModem, 'fsk': FSKModem}
    colors = plt.cm.jet(np.linspace(0,1,len(arrayOfM))) # Colormap
    fig, ax = plt.subplots(nrows=1,ncols=1)

    for i,M in enumerate(arrayOfM):

        #--------Initialization of various parameters--------
        k = np.log2(M)
        EsN0dBs = 10*np.log10(k)*EbN0dBs # EsN0dB calculation
        SER_sim = np.zeros(len(EbN0dBs)) # Simulated symbol error rates
        inputSyms = np.random.randint(low=0, high=M, size=nSym)
        # Uniform random symbols from 0 to M-1

        if mod_type.lower() == 'fsk':
            modem=modem_dict[mod_type.lower()](M,coherence) # Choose modem from dictionary
        else: # For all other modulations
            modem=modem_dict[mod_type.lower()](M) # Choose modem from dictionary
        modulatedSyms = modem.modulate(inputSyms) # Modulate

        
