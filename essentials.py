def plotWelchPSD(x, fs, fc, ax = None, color = 'b', label = None):

    from scipy.signal import welch
    from numpy import log10, hanning
    import matplotlib.pyplot as plt

    nx = max(x.shape)
    na = 16 # averaging factor to plot averaged welch spectrum
    w = hanning(nx//na)
    f, Pxx = welch(x, fs, window = w, noverlap = 0)
    indices = (f>=fc) & (f<4*fc) # to plot PSD from Fc to 4*Fc
    Pxx = Pxx[indices]/Pxx[indices][0] # normalize PSD w.r.t. Fc

    plt.figure(1)
    plt.plot(f[indices]-fc, 10*log10(Pxx))
    plt.title('Welch plot')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.savefig('welch_example_im2.png')

