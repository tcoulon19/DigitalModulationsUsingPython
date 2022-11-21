def modem_demo():

    import numpy as np
    from modem import PSKModem, QAMModem

    M = 16 # 16 points in the constellation
    qamModem = QAMModem(M) # Create a 16-PSK modem object
    qamModem.plotConstellation() # Plot ideal constellation for this modem

    nSym = 10 # 10 symbols as input to PSK modem
    inputSyms = np.random.randint(low=0, high=M, size=nSym) # Uniform random symbols from 0 to M-1
    print('inputSyms: ' + str(inputSyms))

    modulatedSyms = qamModem.modulate(inputSyms) # Modulate
    print('modulatedSyms: ' + str(modulatedSyms))

    detectedSyms = qamModem.demodulate(modulatedSyms) # Demodulate
    print('detectedSyms: ' + str(detectedSyms))

modem_demo()
